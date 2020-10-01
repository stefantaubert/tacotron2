import logging
import os
import random
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from logging import Logger
from typing import Any, Dict, List, Optional
from typing import OrderedDict as OrderedDictType
from typing import Tuple

import numpy as np
import tensorflow as tf
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.core.common.accents_dict import PADDING_ACCENT, AccentsDict
from src.core.common.speakers_dict import SpeakersDict
from src.core.common.symbol_id_dict import PADDING_SYMBOL, SymbolIdDict
from src.core.common.symbols_map import (SymbolsMap, get_map,
                                         symbols_map_to_symbols_ids_map)
from src.core.common.taco_stft import TacotronSTFT
from src.core.common.text import deserialize_list
from src.core.common.train import (SaveIterationSettings, check_save_it,
                                   filter_checkpoints,
                                   get_all_checkpoint_iterations,
                                   get_checkpoint,
                                   get_continue_batch_iteration,
                                   get_continue_epoch,
                                   get_formatted_current_total, hp_from_raw,
                                   hp_raw, overwrite_custom_hparams,
                                   skip_batch)
from src.core.pre.merge_ds import PreparedData, PreparedDataList
from src.core.tacotron.hparams import create_hparams
from src.core.tacotron.logger import Tacotron2Logger
from src.core.tacotron.model import (SPEAKER_EMBEDDING_LAYER_NAME,
                                     SYMBOL_EMBEDDING_LAYER_NAME, Tacotron2,
                                     get_model_symbol_id, get_model_symbol_ids,
                                     get_symbol_weights, update_weights)


@dataclass
class CheckpointTacotron():
  # Renaming of any of these fields will destroy previous models!
  state_dict: dict
  optimizer: dict
  learning_rate: float
  iteration: int
  hparams: dict
  speakers: OrderedDictType[str, int]
  symbols: OrderedDictType[str, int]
  accents: OrderedDictType[str, int]

  def get_symbols(self) -> SymbolIdDict:
    return SymbolIdDict.from_raw(self.symbols)

  def get_accents(self) -> AccentsDict:
    return AccentsDict.from_raw(self.accents)

  def get_speakers(self) -> SpeakersDict:
    return SpeakersDict.from_raw(self.speakers)

  def get_hparams(self) -> tf.contrib.training.HParams:
    return hp_from_raw(self.hparams)

  def save(self, checkpoint_path: str, logger: Logger):
    logger.info(f"Saving model at iteration {self.iteration}...")
    checkpoint_dict = asdict(self)
    torch.save(checkpoint_dict, checkpoint_path)
    logger.info(f"Saved model to '{checkpoint_path}'.")

  def get_symbol_embedding_weights(self) -> torch.Tensor:
    pretrained_weights = self.state_dict[SYMBOL_EMBEDDING_LAYER_NAME]
    return pretrained_weights

  @classmethod
  def load(cls, checkpoint_path: str, logger: Logger):
    assert os.path.isfile(checkpoint_path)
    logger.info(f"Loading tacotron model '{checkpoint_path}'...")
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    result = cls(**checkpoint_dict)
    logger.info(f"Loaded model at iteration {result.iteration}.")
    logger.info(f'Including {len(result.symbols)} symbols.')
    logger.info(f'Including {len(result.accents)} accents.')
    logger.info(f'Including {len(result.speakers)} speaker(s).')
    return result


class SymbolsMelLoader(Dataset):
  """
    1) loads audio,text pairs
    2) normalizes text and converts them to sequences of one-hot vectors
    3) computes mel-spectrograms from audio files.
  """

  def __init__(self, prepare_ds_ms_data: PreparedDataList, hparams, logger: Logger):
    data = prepare_ds_ms_data

    random.seed(hparams.seed)
    random.shuffle(data)
    self.use_saved_mels: bool = hparams.use_saved_mels
    if not hparams.use_saved_mels:
      self.mel_parser = TacotronSTFT.fromhparams(hparams)

    logger.info("Reading files...")
    self.data: Dict[int, Tuple[torch.IntTensor, torch.IntTensor, str, int]] = {}
    values: PreparedData
    for i, values in enumerate(tqdm(data.items())):
      symbol_ids = deserialize_list(values.serialized_symbol_ids)
      accent_ids = deserialize_list(values.serialized_accent_ids)

      model_symbol_ids = get_model_symbol_ids(
        symbol_ids, accent_ids, hparams.n_symbols, hparams.accents_use_own_symbols)

      symbols_tensor = torch.IntTensor(model_symbol_ids)
      accents_tensor = torch.IntTensor(accent_ids)

      if hparams.use_saved_mels:
        self.data[i] = (symbols_tensor, accents_tensor, values.mel_path, values.speaker_id)
      else:
        self.data[i] = (symbols_tensor, accents_tensor, values.wav_path, values.speaker_id)

    if hparams.use_saved_mels and hparams.cache_mels:
      logger.info("Loading mels into memory...")
      self.cache: Dict[int, torch.Tensor] = {}
      vals: tuple
      for i, vals in tqdm(self.data.items()):
        mel_tensor = torch.load(vals[1], map_location='cpu')
        self.cache[i] = mel_tensor
    self.use_cache: bool = hparams.cache_mels

  def __getitem__(self, index: int) -> Tuple[torch.IntTensor, torch.IntTensor, torch.Tensor, int]:
    # return self.cache[index]
    # debug_logger.debug(f"getitem called {index}")
    symbols_tensor, accents_tensor, path, speaker_id = self.data[index]
    if self.use_saved_mels:
      if self.use_cache:
        mel_tensor = self.cache[index].clone().detach()
      else:
        mel_tensor: torch.Tensor = torch.load(path, map_location='cpu')
    else:
      mel_tensor = self.mel_parser.get_mel_tensor_from_file(path)

    symbols_tensor_cloned = symbols_tensor.clone().detach()
    accents_tensor_cloned = accents_tensor.clone().detach()
    # debug_logger.debug(f"getitem finished {index}")
    return symbols_tensor_cloned, accents_tensor_cloned, mel_tensor, speaker_id

  def __len__(self):
    return len(self.data)


class SymbolsMelCollate():
  """ Zero-pads model inputs and targets based on number of frames per step
  """

  def __init__(self, n_frames_per_step: int, padding_symbol_id: int, padding_accent_id: int):
    self.n_frames_per_step = n_frames_per_step
    self.padding_symbol_id = padding_symbol_id
    self.padding_accent_id = padding_accent_id

  def __call__(self, batch: List[Tuple[torch.IntTensor, torch.IntTensor, torch.Tensor, int]]):
    """Collate's training batch from normalized text and mel-spectrogram
    PARAMS
    ------
    batch: [text_normalized, mel_normalized]
    """
    # Right zero-pad all one-hot text sequences to max input length
    input_lengths, ids_sorted_decreasing = torch.sort(
      torch.LongTensor([len(symbols_tensor) for symbols_tensor, _, _, _ in batch]), dim=0, descending=True)
    max_input_len = input_lengths[0]

    symbols_padded = torch.LongTensor(len(batch), max_input_len)
    torch.nn.init.constant_(symbols_padded, self.padding_symbol_id)

    accents_padded = torch.LongTensor(len(batch), max_input_len)
    torch.nn.init.constant_(accents_padded, self.padding_accent_id)

    for i, batch_id in enumerate(ids_sorted_decreasing):
      symbols = batch[batch_id][0]
      symbols_padded[i, :symbols.size(0)] = symbols

      accents = batch[batch_id][1]
      accents_padded[i, :accents.size(0)] = accents

    # Right zero-pad mel-spec
    _, _, first_mel, _ = batch[0]
    num_mels = first_mel.size(0)
    max_target_len = max([mel_tensor.size(1) for _, _, mel_tensor, _ in batch])
    if max_target_len % self.n_frames_per_step != 0:
      max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
      assert max_target_len % self.n_frames_per_step == 0

    # include mel padded and gate padded
    mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
    mel_padded.zero_()

    gate_padded = torch.FloatTensor(len(batch), max_target_len)
    gate_padded.zero_()

    output_lengths = torch.LongTensor(len(batch))
    for i, batch_id in enumerate(ids_sorted_decreasing):
      _, _, mel, _ = batch[batch_id]
      mel_padded[i, :, :mel.size(1)] = mel
      gate_padded[i, mel.size(1) - 1:] = 1
      output_lengths[i] = mel.size(1)

    # count number of items - characters in text
    # len_x = []
    speaker_ids = []
    for i, batch_id in enumerate(ids_sorted_decreasing):
      # len_symb = batch[batch_id][0].get_shape()[0]
      # len_x.append(len_symb)
      _, _, _, speaker_id = batch[batch_id]
      speaker_ids.append(speaker_id)

    # len_x = torch.Tensor(len_x)
    speaker_ids = torch.LongTensor(speaker_ids)

    return Tacotron2.make_batch(
      symbols_padded,
      accents_padded,
      input_lengths,
      mel_padded,
      gate_padded,
      output_lengths,
      speaker_ids
    )


# import torch.distributed as dist

# def reduce_tensor(tensor, n_gpus):
#   rt = tensor.clone()
#   dist.all_reduce(rt, op=dist.reduce_op.SUM)
#   rt /= n_gpus
#   return rt

# def init_distributed(hparams, n_gpus, rank, group_name, training_dir_path):
#   assert torch.cuda.is_available(), "Distributed mode requires CUDA."
#   log(training_dir_path, "Initializing Distributed")

#   # Set cuda device so everything is done on the right GPU.
#   torch.cuda.set_device(rank % torch.cuda.device_count())

#   # Initialize distributed communication
#   dist.init_process_group(
#     backend=hparams.dist_backend, init_method=hparams.dist_url,
#     world_size=n_gpus, rank=rank, group_name=group_name)

#   log(training_dir_path, "Done initializing distributed")


class Tacotron2Loss(nn.Module):
  def forward(self, model_output, targets) -> torch.Tensor:
    mel_target, gate_target = targets[0], targets[1]
    mel_target.requires_grad = False
    gate_target.requires_grad = False
    gate_target = gate_target.view(-1, 1)

    mel_out, mel_out_postnet, gate_out, _ = model_output
    gate_out = gate_out.view(-1, 1)
    mel_loss = nn.MSELoss()(mel_out, mel_target) + \
        nn.MSELoss()(mel_out_postnet, mel_target)
    gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
    return mel_loss + gate_loss


def prepare_valloader(hparams, collate_fn: SymbolsMelCollate, valset: PreparedDataList, logger: Logger) -> DataLoader:
  logger.info(
    f"Duration valset {valset.get_total_duration_s() / 60:.2f}min / {valset.get_total_duration_s() / 60 / 60:.2f}h")

  val = SymbolsMelLoader(valset, hparams, logger)
  val_sampler = None

  # if distributed_run:
  #   val_sampler = DistributedSampler(val)

  val_loader = DataLoader(
    dataset=val,
    sampler=val_sampler,
    num_workers=1,
    shuffle=False,
    batch_size=hparams.batch_size,
    pin_memory=False,
    collate_fn=collate_fn
  )

  return val_loader


def prepare_trainloader(hparams, collate_fn: SymbolsMelCollate, trainset: PreparedDataList, logger: Logger) -> DataLoader:
  # Get data, data loaders and collate function ready
  logger.info(
    f"Duration trainset {trainset.get_total_duration_s() / 60:.2f}min / {trainset.get_total_duration_s() / 60 / 60:.2f}h")

  trn = SymbolsMelLoader(trainset, hparams, logger)

  train_sampler = None
  shuffle = True

  # if hparams.distributed_run:
  #   train_sampler = DistributedSampler(trn)
  #   shuffle = False

  train_loader = DataLoader(
    dataset=trn,
    num_workers=1,
    shuffle=shuffle,
    sampler=train_sampler,
    batch_size=hparams.batch_size,
    pin_memory=False,
    drop_last=True,
    collate_fn=collate_fn
  )

  return train_loader


def load_model(hparams, state_dict: Optional[dict], logger: logging.Logger):
  model = Tacotron2(hparams, logger).cuda()
  # if hparams.fp16_run:
  #   model.decoder.attention_layer.score_mask_value = finfo('float16').min

  # if hparams.distributed_run:
  #   model = apply_gradient_allreduce(model)
  if state_dict is not None:
    model.load_state_dict(state_dict)

  return model


def validate_core(model: Tacotron2, criterion: nn.Module, val_loader: DataLoader) -> Tuple:
  """Handles all the validation scoring and printing"""
  model.eval()
  res = []
  with torch.no_grad():
    total_val_loss = 0.0
    for batch in tqdm(val_loader):
      x, y = Tacotron2.parse_batch(batch)
      y_pred = model(x)
      loss = criterion(y_pred, y)
      # if distributed_run:
      #   reduced_val_loss = reduce_tensor(loss.data, n_gpus).item()
      # else:
      #  reduced_val_loss = loss.item()
      reduced_val_loss = loss.item()
      res.append((reduced_val_loss, model, y, y_pred))
      total_val_loss += reduced_val_loss
    avg_val_loss = total_val_loss / len(val_loader)
  model.train()

  return avg_val_loss, res


def validate(model: Tacotron2, criterion: nn.Module, val_loader: DataLoader, iteration: int, taco_logger: Tacotron2Logger, logger: Logger):
  logger.debug("Validating...")
  avg_val_loss, res = validate_core(model, criterion, val_loader)
  logger.info(f"Validation loss {iteration}: {avg_val_loss:9f}")

  logger.debug("Logging to tensorboard...")
  log_only_last_validation_batch = True
  if log_only_last_validation_batch:
    taco_logger.log_validation(*res[-1], iteration)
  else:
    for entry in tqdm(res):
      taco_logger.log_validation(*entry, iteration)
  logger.debug("Finished.")

  return avg_val_loss


def train(warm_model: Optional[CheckpointTacotron], custom_hparams: Optional[Dict[str, str]], taco_logger: Tacotron2Logger, symbols: SymbolIdDict, speakers: SpeakersDict, accents: AccentsDict, trainset: PreparedDataList, valset: PreparedDataList, save_callback: Any, weights_checkpoint: Optional[CheckpointTacotron], weights_map: Optional[SymbolsMap], logger: Logger, checkpoint_logger: Logger):
  logger.info("Starting new training...")

  _train(
    custom_hparams=custom_hparams,
    taco_logger=taco_logger,
    trainset=trainset,
    valset=valset,
    save_callback=save_callback,
    speakers=speakers,
    accents=accents,
    symbols=symbols,
    weights_checkpoint=weights_checkpoint,
    weights_map=weights_map,
    warm_model=warm_model,
    checkpoint=None,
    logger=logger,
    checkpoint_logger=checkpoint_logger
  )


def continue_train(checkpoint: CheckpointTacotron, custom_hparams: Optional[Dict[str, str]], taco_logger: Tacotron2Logger, trainset: PreparedDataList, valset: PreparedDataList, save_callback: Any, logger: Logger, checkpoint_logger: Logger):
  logger.info("Continuing training...")

  _train(
    custom_hparams=custom_hparams,
    taco_logger=taco_logger,
    trainset=trainset,
    valset=valset,
    save_callback=save_callback,
    speakers=checkpoint.get_speakers(),
    accents=checkpoint.get_accents(),
    symbols=checkpoint.get_symbols(),
    weights_checkpoint=None,
    weights_map=None,
    warm_model=None,
    checkpoint=checkpoint,
    logger=logger,
    checkpoint_logger=checkpoint_logger
  )


def _train(custom_hparams: Optional[Dict[str, str]], taco_logger: Tacotron2Logger, trainset: PreparedDataList, valset: PreparedDataList, save_callback: Any, speakers: SpeakersDict, accents: AccentsDict, symbols: SymbolIdDict, checkpoint: Optional[CheckpointTacotron], warm_model: Optional[CheckpointTacotron], weights_checkpoint: Optional[CheckpointTacotron], weights_map: SymbolsMap, logger: Logger, checkpoint_logger: Logger):
  """Training and validation logging results to tensorboard and stdout
  Params
  ------
  output_directory (string): directory to save checkpoints
  log_directory (string) directory to save tensorboard logs
  checkpoint_path(string): checkpoint path
  n_gpus (int): number of gpus
  rank (int): rank of current gpu
  hparams (object): comma separated list of "name=value" pairs.
  """

  complete_start = time.time()

  if checkpoint is not None:
    hparams = checkpoint.get_hparams()
  else:
    hparams = create_hparams(
      n_accents=len(accents),
      n_speakers=len(speakers),
      n_symbols=len(symbols)
    )
  # is it problematic to change the batch size?
  overwrite_custom_hparams(hparams, custom_hparams)

  logger.info('Final parsed hparams:')
  logger.info('\n'.join(str(hparams.values()).split(',')))

  torch.manual_seed(hparams.seed)
  torch.cuda.manual_seed(hparams.seed)
  torch.backends.cudnn.enabled = hparams.cudnn_enabled
  torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

  if checkpoint is not None:
    model, optimizer, learning_rate, iteration = model_and_optimizer_from_checkpoint(
      checkpoint, hparams, logger)
  else:
    model, optimizer, learning_rate, iteration = model_and_optimizer_fresh(hparams, logger)

    if warm_model is not None:
      warm_start_model(model, warm_model.state_dict, hparams.ignore_layers, logger)

    if weights_checkpoint is not None:
      pretrained_weights = get_mapped_symbol_weights(
        model_symbols=symbols,
        trained_weights=weights_checkpoint.get_symbol_embedding_weights(),
        trained_symbols=weights_checkpoint.get_symbols(),
        custom_mapping=weights_map,
        hparams=hparams,
        logger=logger
      )

      update_weights(model.embedding, pretrained_weights)

  logger.debug("Modelweights:")
  logger.debug(f"is cuda: {model.embedding.weight.is_cuda}")
  logger.debug(str(model.state_dict()[SYMBOL_EMBEDDING_LAYER_NAME]))

  collate_fn = SymbolsMelCollate(
    n_frames_per_step=hparams.n_frames_per_step,
    padding_symbol_id=symbols.get_id(PADDING_SYMBOL),
    padding_accent_id=accents.get_id(PADDING_ACCENT)
  )

  val_loader = prepare_valloader(hparams, collate_fn, valset, logger)
  train_loader = prepare_trainloader(hparams, collate_fn, trainset, logger)

  batch_iterations = len(train_loader)
  if batch_iterations == 0:
    logger.error("Not enough trainingdata.")
    raise Exception()

  save_it_settings = SaveIterationSettings(
    epochs=hparams.epochs,
    batch_iterations=batch_iterations,
    save_first_iteration=True,
    save_last_iteration=True,
    iters_per_checkpoint=hparams.iters_per_checkpoint,
    epochs_per_checkpoint=hparams.epochs_per_checkpoint
  )

  criterion = Tacotron2Loss()
  batch_durations: List[float] = []

  train_start = time.perf_counter()
  start = train_start
  model.train()
  continue_epoch = get_continue_epoch(iteration, batch_iterations)
  for epoch in range(continue_epoch, hparams.epochs):
    next_batch_iteration = get_continue_batch_iteration(iteration, batch_iterations)
    skip_bar = None
    if next_batch_iteration > 0:
      logger.debug(f"Current batch is {next_batch_iteration} of {batch_iterations}")
      logger.debug("Skipping batches...")
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

      for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

      model.zero_grad()
      x, y = Tacotron2.parse_batch(batch)
      y_pred = model(x)

      loss = criterion(y_pred, y)
      # if hparams.distributed_run:
      #   reduced_loss = reduce_tensor(loss.data, n_gpus).item()
      # else:
      #   reduced_loss = loss.item()
      reduced_loss = loss.item()

      # if hparams.fp16_run:
      #   with amp.scale_loss(loss, optimizer) as scaled_loss:
      #     scaled_loss.backward()
      # else:
      #   loss.backward()
      loss.backward()

      # if hparams.fp16_run:
      #   grad_norm = torch.nn.utils.clip_grad_norm_(
      #     amp.master_params(optimizer), hparams.grad_clip_thresh)
      #   is_overflow = math.isnan(grad_norm)
      # else:
      #   grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hparams.grad_clip_thresh)
      grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hparams.grad_clip_thresh)

      optimizer.step()

      iteration += 1

      end = time.perf_counter()
      duration = end - start
      start = end

      batch_durations.append(duration)
      logger.info(" | ".join([
        f"Epoch: {get_formatted_current_total(epoch + 1, hparams.epochs)}",
        f"Iteration: {get_formatted_current_total(batch_iteration + 1, batch_iterations)}",
        f"Total iteration: {get_formatted_current_total(iteration, hparams.epochs * batch_iterations)}",
        f"Train loss: {reduced_loss:.6f}",
        f"Grad Norm: {grad_norm:.6f}",
        f"Duration: {duration:.2f}s/it",
        f"Avg. duration: {np.mean(batch_durations):.2f}s/it",
        f"Total Duration: {(time.perf_counter() - train_start) / 60 / 60:.2f}h"
      ]))

      taco_logger.log_training(reduced_loss, grad_norm, learning_rate,
                               duration, iteration)

      save_it = check_save_it(epoch, iteration, save_it_settings)
      if save_it:
        checkpoint = CheckpointTacotron(
          state_dict=model.state_dict(),
          optimizer=optimizer.state_dict(),
          learning_rate=learning_rate,
          iteration=iteration,
          hparams=hp_raw(hparams),
          symbols=symbols.raw(),
          accents=accents.raw(),
          speakers=speakers.raw()
        )

        save_callback(checkpoint)

        valloss = validate(model, criterion, val_loader, iteration, taco_logger, logger)
        # if rank == 0:
        log_checkpoint_score(iteration, grad_norm,
                             reduced_loss, valloss, epoch, batch_iteration, checkpoint_logger)

  duration_s = time.time() - complete_start
  logger.info(f'Finished training. Total duration: {duration_s / 60:.2f}min')


def update_symbol_embeddings(model: Tacotron2, weights: torch.Tensor, logger: Logger):
  logger.info("Loading pretrained mapped embeddings...")
  update_weights(model.symbol_embedding, weights)


def model_and_optimizer_fresh(hparams, logger: Logger):
  logger.info("Starting new model...")
  model = load_model(hparams, None, logger)

  # if hparams.distributed_run:
  #   init_distributed(hparams, n_gpus, rank, group_name, training_dir_path)

  # if hparams.fp16_run:
  #   from apex import amp
  #   model, optimizer = amp.initialize(model, optimizer, opt_level='O2')

  # if hparams.distributed_run:
  #   model = apply_gradient_allreduce(model)

  optimizer = torch.optim.Adam(
    params=model.parameters(),
    lr=hparams.learning_rate,
    weight_decay=hparams.weight_decay
  )

  current_iteration = 0

  return model, optimizer, hparams.learning_rate, current_iteration


def model_and_optimizer_from_checkpoint(checkpoint: CheckpointTacotron, updated_hparams, logger: Logger):
  logger.info("Continuing training from checkpoint...")
  learning_rate: float = updated_hparams.learning_rate
  if updated_hparams.use_saved_learning_rate:
    learning_rate = checkpoint.learning_rate

  model = load_model(updated_hparams, checkpoint.state_dict, logger)

  # warn: use saved learning rate is ignored here
  optimizer = torch.optim.Adam(
    params=model.parameters(),
    lr=learning_rate,
    weight_decay=updated_hparams.weight_decay
  )
  optimizer.load_state_dict(checkpoint.optimizer)

  return model, optimizer, learning_rate, checkpoint.iteration


def warm_start_model(model: Tacotron2, warm_start_states: dict, ignore_layers: List[str], logger: Logger):
  logger.info("Loading states from pretrained model...")
  # The default value from HParams is [""], an empty list was not working.
  ignore_layers.extend([
    SYMBOL_EMBEDDING_LAYER_NAME,
    # ACCENT_EMBEDDING_LAYER_NAME,
    SPEAKER_EMBEDDING_LAYER_NAME
  ])

  model_dict = {k: v for k, v in warm_start_states.items() if k not in ignore_layers}
  _update_model_state_dict(model, model_dict)


def _update_model_state_dict(model: Tacotron2, updates: dict):
  dummy_dict = model.state_dict()
  dummy_dict.update(updates)
  model.load_state_dict(dummy_dict)


def log_checkpoint_score(iteration: int, gradloss: float, trainloss: float, valloss: float, epoch: int, i: int, checkpoint_logger: Logger):
  loss_avg = (trainloss + valloss) / 2
  msg = f"{iteration}\tepoch-{epoch}\tit-{i}\tgradloss-{gradloss:.6f}\ttrainloss-{trainloss:.6f}\tvalidationloss-{valloss:.6f}\tavg-train-val-{loss_avg:.6f}"
  checkpoint_logger.info(msg)


def symbols_ids_map_to_model_symbols_ids_map(symbols_id_map: OrderedDictType[int, int], n_accents: int, n_symbols: int, accents_use_own_symbols: bool) -> OrderedDictType[int, int]:
  res: OrderedDictType[int, int] = OrderedDict()

  for accent_id in range(n_accents):
    for map_to_symbol_id, map_from_symbol_id in symbols_id_map.items():

      map_to_model_id = get_model_symbol_id(
        map_to_symbol_id,
        accent_id,
        n_symbols,
        accents_use_own_symbols
      )

      res[map_to_model_id] = symbols_id_map[map_from_symbol_id]

    if not accents_use_own_symbols:
      break

  return res


def map_weights(model_symbols_id_map: OrderedDictType[int, int], model_weights, trained_weights, logger: Logger):
  for map_to_model_symbol_id, map_from_symbol_id in model_symbols_id_map.items():
    assert 0 <= map_to_model_symbol_id < model_weights.shape[0]
    assert 0 <= map_from_symbol_id < trained_weights.shape[0]

    logger.debug(f"Mapped {map_from_symbol_id} to {map_to_model_symbol_id}.")
    model_weights[map_to_model_symbol_id] = trained_weights[map_from_symbol_id]


def get_mapped_symbol_weights(model_symbols: SymbolIdDict, trained_weights: torch.Tensor, trained_symbols: SymbolIdDict, custom_mapping: Optional[SymbolsMap], hparams, logger: Logger) -> torch.Tensor:
  symbols_match_not_model = trained_weights.shape[0] != len(trained_symbols)
  if symbols_match_not_model:
    logger.exception(
      f"Weights mapping: symbol space from pretrained model ({trained_weights.shape[0]}) did not match amount of symbols ({len(trained_symbols)}).")
    raise Exception()

  symbols_map = get_map(
    dest_symbols=model_symbols.get_all_symbols(),
    orig_symbols=trained_symbols.get_all_symbols(),
    symbols_mapping=custom_mapping,
    logger=logger
  )

  symbols_id_map = symbols_map_to_symbols_ids_map(
    dest_symbols=model_symbols,
    orig_symbols=trained_symbols,
    symbols_mapping=symbols_map,
    logger=logger
  )

  model_symbols_id_map = symbols_ids_map_to_model_symbols_ids_map(
    symbols_id_map,
    hparams.n_accents,
    n_symbols=hparams.n_symbols,
    accents_use_own_symbols=hparams.accents_use_own_symbols
  )

  model_weights = get_symbol_weights(hparams)

  map_weights(
    model_symbols_id_map=model_symbols_id_map,
    model_weights=model_weights,
    trained_weights=trained_weights,
    logger=logger
  )

  symbols_wo_mapping = symbols_map.get_symbols_without_mapping()
  not_existing_symbols = model_symbols.get_all_symbols() - symbols_map.keys()
  no_mapping = symbols_wo_mapping | not_existing_symbols
  if len(no_mapping) > 0:
    logger.warning(f"Following symbols were not mapped: {no_mapping}")
  else:
    logger.info("All symbols were mapped.")

  return model_weights


def eval_checkpoints(custom_hparams: Optional[str], checkpoint_dir: str, select: int, min_it: int, max_it: int, n_symbols: int, n_accents: int, n_speakers: int, valset: PreparedDataList, logger: Logger):
  its = get_all_checkpoint_iterations(checkpoint_dir)
  logger.info(f"Available iterations {its}")
  filtered_its = filter_checkpoints(its, select, min_it, max_it)
  if len(filtered_its) > 0:
    logger.info(f"Selected iterations: {filtered_its}")
  else:
    logger.info("None selected. Exiting.")
    return

  hparams = create_hparams(n_speakers, n_symbols, n_accents, custom_hparams)

  collate_fn = SymbolsMelCollate(
    hparams.n_frames_per_step,
    padding_symbol_id=0,  # TODO: refactor
    padding_accent_id=0  # TODO: refactor
    # padding_symbol_id=symbols.get_id(PADDING_SYMBOL),
    # padding_accent_id=accents.get_id(PADDING_ACCENT)
  )
  val_loader = prepare_valloader(hparams, collate_fn, valset, logger)

  result = []
  for checkpoint_iteration in tqdm(filtered_its):
    criterion = Tacotron2Loss()
    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)
    full_checkpoint_path, _ = get_checkpoint(checkpoint_dir, checkpoint_iteration)
    state_dict = CheckpointTacotron.load(full_checkpoint_path, logger).state_dict
    model = load_model(hparams, state_dict, logger)
    val_loss, _ = validate_core(model, criterion, val_loader)
    result.append((checkpoint_iteration, val_loss))
    logger.info(f"Validation loss {checkpoint_iteration}: {val_loss:9f}")

  logger.info("Result...")
  logger.info("Sorted after checkpoints:")

  result.sort()
  for cp, loss in result:
    logger.info(f"Validation loss {cp}: {loss:9f}")

  result = [(b, a) for a, b in result]
  result.sort()

  logger.info("Sorted after scores:")
  for loss, cp in result:
    logger.info(f"Validation loss {cp}: {loss:9f}")
