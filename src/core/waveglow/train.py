import os
import random
import time
from dataclasses import asdict, dataclass
from logging import Logger
from typing import Any, List, Optional, Tuple

import numpy as np
import tensorflow as tf
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.core.common.audio import get_wav_tensor_segment, wav_to_float32_tensor
from src.core.common.taco_stft import TacotronSTFT
from src.core.common.train import (SaveIterationSettings, check_save_it,
                                   get_continue_batch_iteration,
                                   get_continue_epoch,
                                   get_formatted_current_total,
                                   get_last_checkpoint, get_pytorch_filename,
                                   hp_from_raw, hp_raw, skip_batch)
from src.core.pre.merge_ds import PreparedDataList
from src.core.waveglow.hparams import create_hparams
from src.core.waveglow.logger import WaveglowLogger
from src.core.waveglow.model import WaveGlow, WaveGlowLoss


@dataclass
class CheckpointWaveglow():
  # Renaming of any of these fields will destroy previous models!
  state_dict: dict
  optimizer: dict
  learning_rate: float
  iteration: int
  hparams: dict

  def get_hparams(self) -> tf.contrib.training.HParams:
    return hp_from_raw(self.hparams)

  def save(self, checkpoint_path: str, logger: Logger):
    logger.info(f"Saving model at iteration {self.iteration}...")
    checkpoint_dict = asdict(self)
    torch.save(checkpoint_dict, checkpoint_path)
    logger.info(f"Saved model to '{checkpoint_path}'.")

  @classmethod
  def load(cls, checkpoint_path: str, logger: Logger):
    assert os.path.isfile(checkpoint_path)
    logger.info(f"Loading waveglow model '{checkpoint_path}'...")
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    result = cls(**checkpoint_dict)
    logger.info(f"Loaded model at iteration {result.iteration}.")
    return result


class MelLoader(Dataset):
  """
  This is the main class that calculates the spectrogram and returns the
  spectrogram, audio pair.
  """

  def __init__(self, prepare_ds_data: PreparedDataList, hparams, logger: Logger):
    self.taco_stft = TacotronSTFT.fromhparams(hparams)
    self.segment_length: int = hparams.segment_length
    self.sampling_rate: int = hparams.sampling_rate
    self.cache_wavs: bool = hparams.cache_wavs
    self._logger = logger

    data = prepare_ds_data
    random.seed(hparams.seed)
    random.shuffle(data)

    wav_paths = {}
    for i, values in enumerate(data.items()):
      wav_paths[i] = values.wav_path
    self.wav_paths = wav_paths

    if hparams.cache_wavs:
      self._logger.info("Loading wavs into memory...")
      cache = {}
      for i, wav_path in tqdm(wav_paths.items()):
        cache[i] = self._load_wav_tensor(wav_path)
      self._logger.info("Done")
      self.cache = cache

  def _load_wav_tensor(self, wav_path: str):
    wav_tensor, sr = wav_to_float32_tensor(wav_path)
    if sr != self.sampling_rate:
      self._logger.exception(f"{wav_path} {sr} SR doesn't match target {self.sampling_rate} SR")
      raise ValueError()
    return wav_tensor

  def __getitem__(self, index):
    if self.cache_wavs:
      wav_tensor = self.cache[index].clone().detach()
    else:
      wav_tensor = self._load_wav_tensor(self.wav_paths[index])
    wav_tensor = get_wav_tensor_segment(wav_tensor, self.segment_length)
    mel_tensor = self.taco_stft.get_mel_tensor(wav_tensor)
    return (mel_tensor, wav_tensor)

  def __len__(self):
    return len(self.wav_paths)

# #=====START: ADDED FOR DISTRIBUTED======
# from distributed_waveglow import init_distributed, apply_gradient_allreduce, reduce_tensor
# from torch.utils.data.distributed import DistributedSampler
# #=====END:   ADDED FOR DISTRIBUTED======


def load_model(hparams, state_dict: Optional[dict]):
  model = WaveGlow(hparams)
  model = model.cuda()

  if state_dict is not None:
    model.load_state_dict(state_dict)

  return model


def prepare_dataloaders(hparams, trainset: PreparedDataList, valset: PreparedDataList, logger: Logger):
  logger.info(
    f"Duration trainset {trainset.get_total_duration_s() / 60:.2f}min / {trainset.get_total_duration_s() / 60 / 60:.2f}h")
  logger.info(
    f"Duration valset {valset.get_total_duration_s() / 60:.2f}min / {valset.get_total_duration_s() / 60 / 60:.2f}h")

  trn = MelLoader(trainset, hparams, logger)

  train_sampler = None
  shuffle = False  # maybe set to true bc taco is also true

  # # =====START: ADDED FOR DISTRIBUTED======
  # train_sampler = DistributedSampler(trn) if n_gpus > 1 else None
  # # =====END:   ADDED FOR DISTRIBUTED======
  train_loader = DataLoader(
    dataset=trn,
    num_workers=0,
    shuffle=shuffle,
    sampler=train_sampler,
    batch_size=hparams.batch_size,
    pin_memory=False,
    drop_last=True
  )

  val = MelLoader(valset, hparams, logger)
  val_sampler = None

  val_loader = DataLoader(
    dataset=val,
    sampler=val_sampler,
    num_workers=0,
    shuffle=False,
    batch_size=hparams.batch_size,
    pin_memory=False
  )

  return train_loader, val_loader


def validate_core(model, criterion, val_loader: DataLoader, logger: Logger):
  """Handles all the validation scoring and printing"""
  model.eval()
  with torch.no_grad():

    # if distributed_run:
    #   val_sampler = DistributedSampler(valset)

    val_loss = 0.0
    logger.info("Validating...")
    for batch in tqdm(val_loader):
      x = model.parse_batch(batch)
      y_pred = model(x)
      loss = criterion(y_pred)
      # if distributed_run:
      #   reduced_val_loss = reduce_tensor(loss.data, n_gpus).item()
      # else:
      #  reduced_val_loss = loss.item()
      reduced_val_loss = loss.item()
      val_loss += reduced_val_loss
    avg_val_loss = val_loss / len(val_loader)

  model.train()
  return avg_val_loss, model, y_pred


def validate(model, criterion, val_loader, iteration, logger: WaveglowLogger, debug_logger: Logger):
  val_loss, model, y_pred = validate_core(model, criterion, val_loader, debug_logger)
  debug_logger.info("Validation loss {}: {:9f}".format(iteration, val_loss))
  logger.log_validation(val_loss, model, y_pred, iteration)

  return val_loss


def continue_train(custom_hparams: str, logdir: str, trainset: PreparedDataList, valset: PreparedDataList, save_checkpoint_dir: str, debug_logger):
  debug_logger.info("Continuing training...")
  last_checkpoint_path, _ = get_last_checkpoint(save_checkpoint_dir)

  _train(
    custom_hparams=custom_hparams,
    logdir=logdir,
    trainset=trainset,
    valset=valset,
    save_checkpoint_dir=save_checkpoint_dir,
    checkpoint=CheckpointWaveglow.load(last_checkpoint_path, debug_logger),
    debug_logger=debug_logger
  )


def train(custom_hparams: str, logdir: str, trainset: PreparedDataList, valset: PreparedDataList, save_checkpoint_dir: str, debug_logger):
  debug_logger.info("Starting new training...")

  _train(
    custom_hparams=custom_hparams,
    logdir=logdir,
    trainset=trainset,
    valset=valset,
    save_checkpoint_dir=save_checkpoint_dir,
    checkpoint=None,
    debug_logger=debug_logger
  )


def _train(custom_hparams: str, logdir: str, trainset: PreparedDataList, valset: PreparedDataList, save_checkpoint_dir: str, checkpoint: Optional[CheckpointWaveglow], debug_logger: Logger):
  complete_start = time.time()
  logger = WaveglowLogger(logdir)

  # if hparams.distributed_run:
  #   init_distributed(hparams, n_gpus, rank, group_name, training_dir_path)

  # rank = 0 # 'rank of current gpu'
  # n_gpus = torch.cuda.device_count() # 'number of gpus'
  # if n_gpus > 1:
  #   raise Exception("More than one GPU is currently not supported.")
  # group_name = "group_name" # 'Distributed group name'
  # if n_gpus > 1:
  #   if args.group_name == '':
  #     print("WARNING: Multiple GPUs detected but no distributed group set")
  #     print("Only running 1 GPU.  Use distributed.py for multiple GPUs")
  #     n_gpus = 1
  # if n_gpus == 1 and args.rank != 0:
  #   raise Exception("Doing single GPU training on rank > 0")

  # if hparams.fp16_run:
  #   from apex import amp
  #   model, optimizer = amp.initialize(model, optimizer, opt_level='O2')

  # if hparams.distributed_run:
  #   model = apply_gradient_allreduce(model)

  if checkpoint is not None:
    model, optimizer, iteration, hparams = model_and_optimizer_from_checkpoint(
      checkpoint,
      custom_hparams,
      debug_logger
    )
  else:
    model, optimizer, iteration, hparams = model_and_optimizer_fresh(custom_hparams, debug_logger)

  debug_logger.info('Final parsed hparams:')
  debug_logger.info('\n'.join(str(hparams.values()).split(',')))

  debug_logger.info("Distributed Run: {}".format(False))
  debug_logger.info("cuDNN Enabled: {}".format(torch.backends.cudnn.enabled))
  debug_logger.info("cuDNN Benchmark: {}".format(torch.backends.cudnn.benchmark))

  torch.backends.cudnn.enabled = True
  torch.backends.cudnn.benchmark = False

  torch.manual_seed(hparams.seed)
  torch.cuda.manual_seed(hparams.seed)

  # #=====START: ADDED FOR DISTRIBUTED======
  # if n_gpus > 1:
  #   model = apply_gradient_allreduce(model)
  # #=====END:   ADDED FOR DISTRIBUTED======

  # if fp16_run:
  #   from apex import amp
  #   model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

  # #=====START: ADDED FOR DISTRIBUTED======
  # if n_gpus > 1:
  #   init_distributed(rank, n_gpus, group_name, **dist_config)
  # #=====END:   ADDED FOR DISTRIBUTED======

  criterion = WaveGlowLoss(
    sigma=hparams.sigma
  )

  train_loader, val_loader = prepare_dataloaders(
    hparams,
    trainset,
    valset,
    debug_logger
  )

  batch_iterations = len(train_loader)
  if batch_iterations == 0:
    debug_logger.error("Not enough trainingdata.")
    raise Exception()

  # Get shared output_directory ready
  # if rank == 0:
  #   if not os.path.isdir(output_directory):
  #     os.makedirs(output_directory)
  #     os.chmod(output_directory, 0o775)
  #   print("output directory", output_directory)

  # if hparams.with_tensorboard and rank == 0:

  model.train()

  train_start = time.perf_counter()
  start = train_start

  save_it_settings = SaveIterationSettings(
    epochs=hparams.epochs,
    batch_iterations=batch_iterations,
    save_first_iteration=True,
    save_last_iteration=True,
    iters_per_checkpoint=hparams.iters_per_checkpoint,
    epochs_per_checkpoint=hparams.epochs_per_checkpoint
  )

  # total_its = hparams.epochs * len(train_loader)
  # epoch_offset = max(0, int(iteration / len(train_loader)))
  # # ================ MAIN TRAINING LOOP! ===================
  # for epoch in range(epoch_offset, hparams.epochs):
  #   debug_logger.info("Epoch: {}".format(epoch))
  #   for i, batch in enumerate(train_loader):
  batch_durations: List[float] = []

  continue_epoch = get_continue_epoch(iteration, batch_iterations)
  for epoch in range(continue_epoch, hparams.epochs):
    next_batch_iteration = get_continue_batch_iteration(iteration, batch_iterations)
    skip_bar = None
    if next_batch_iteration > 0:
      debug_logger.debug(f"Current batch is {next_batch_iteration} of {batch_iterations}")
      debug_logger.debug("Skipping batches...")
      skip_bar = tqdm(total=next_batch_iteration)
    for batch_iteration, batch in enumerate(train_loader):
      need_to_skip_batch = skip_batch(
        batch_iteration=batch_iteration,
        continue_batch_iteration=next_batch_iteration
      )
      if need_to_skip_batch:
        assert skip_bar is not None
        skip_bar.update(1)
        #debug_logger.debug(f"Skipped batch {batch_iteration + 1}/{next_batch_iteration + 1}.")
        continue
      # debug_logger.debug(f"Current batch: {batch[0][0]}")

      model.zero_grad()
      x = WaveGlow.parse_batch(batch)
      y_pred = model(x)

      loss = criterion(y_pred)
      # if n_gpus > 1:
      #   reduced_loss = reduce_tensor(loss.data, n_gpus).item()
      # else:
      #   reduced_loss = loss.item()
      reduced_loss = loss.item()

      # if fp16_run:
      #   with amp.scale_loss(loss, optimizer) as scaled_loss:
      #     scaled_loss.backward()
      # else:
      #   loss.backward()
      loss.backward()

      optimizer.step()

      iteration += 1

      end = time.perf_counter()
      duration = end - start
      start = end

      batch_durations.append(duration)
      debug_logger.info(" | ".join([
        f"Epoch: {get_formatted_current_total(epoch + 1, hparams.epochs)}",
        f"Iteration: {get_formatted_current_total(batch_iteration + 1, batch_iterations)}",
        f"Total iteration: {get_formatted_current_total(iteration, hparams.epochs * batch_iterations)}",
        f"Train loss: {reduced_loss:.6f}",
        f"Duration: {duration:.2f}s/it",
        f"Avg. duration: {np.mean(batch_durations):.2f}s/it",
        f"Total Duration: {(time.perf_counter() - train_start) / 60 / 60:.2f}h"
      ]))

      logger.log_training(reduced_loss, hparams.learning_rate, duration, iteration)

      # if hparams.with_tensorboard and rank == 0:
      logger.add_scalar('training_loss', reduced_loss, iteration)

      # if rank == 0:
      save_it = check_save_it(epoch, iteration, save_it_settings)
      if save_it:
        checkpoint = CheckpointWaveglow(
          state_dict=model.state_dict(),
          optimizer=optimizer.state_dict(),
          learning_rate=hparams.learning_rate,
          iteration=iteration,
          hparams=hp_raw(hparams),
        )

        checkpoint_path = os.path.join(
          save_checkpoint_dir, get_pytorch_filename(iteration))
        checkpoint.save(checkpoint_path, debug_logger)

        validate(model, criterion, val_loader, iteration, logger, debug_logger)

  duration_s = time.time() - complete_start
  debug_logger.info(f'Finished training. Total duration: {duration_s / 60:.2f}min')


def load_optimizer(model: WaveGlow, state_dict: Optional[dict], hparams):
  # warn: use saved learning rate is ignored here

  optimizer = torch.optim.Adam(
    params=model.parameters(),
    lr=hparams.learning_rate,
  )

  if state_dict is not None:
    optimizer.load_state_dict(state_dict)

  return optimizer


def model_and_optimizer_fresh(custom_hparams: str, debug_logger: Logger):
  debug_logger.info("Starting new model...")

  hparams = create_hparams(custom_hparams)
  model = load_model(hparams, None)
  optimizer = load_optimizer(model, None, hparams)
  current_iteration = 0

  return model, optimizer, current_iteration, hparams


def model_and_optimizer_from_checkpoint(checkpoint: CheckpointWaveglow, custom_hparams: str, debug_logger: Logger) -> Tuple[WaveGlow, Any, int]:
  debug_logger.info("Continuing training from checkpoint...")

  updated_hparams = checkpoint.get_hparams()
  # todo apply custom_hparams
  # assert hparams.batch_size == custom.batch_size

  model = load_model(updated_hparams, checkpoint.state_dict)
  optimizer = load_optimizer(model, checkpoint.optimizer, updated_hparams)

  return model, optimizer, checkpoint.iteration, updated_hparams
