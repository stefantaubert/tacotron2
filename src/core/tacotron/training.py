import logging
import os
import random
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from logging import Logger
from math import floor
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

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
                                   get_continue_batch_iteration,
                                   get_continue_epoch, get_custom_checkpoint,
                                   get_formatted_current_total,
                                   get_last_checkpoint, get_pytorch_filename,
                                   skip_batch)
from src.core.pre.merge_ds import PreparedData, PreparedDataList
from src.core.pre.text.pre import OrderedDictType
from src.core.tacotron.hparams import create_hparams
from src.core.tacotron.logger import Tacotron2Logger
from src.core.tacotron.model import (SPEAKER_EMBEDDINGS_LAYER_NAME,
                                     SYMBOL_EMBEDDINGS_LAYER_NAME, Tacotron2,
                                     get_model_symbol_id, get_model_symbol_ids,
                                     get_symbol_weights, update_weights)


@dataclass
class Checkpoint():
  # Renaming of any of these fields will destroy previous models!
  state_dict: dict
  optimizer: dict
  learning_rate: float
  iteration: int
  hparams: tf.contrib.training.HParams
  speakers: SpeakersDict
  symbols: SymbolIdDict
  accents: AccentsDict


def get_train_logger():
  return logging.getLogger("taco-train")


def get_checkpoints_eval_logger():
  return logging.getLogger("taco-train-checkpoints")


debug_logger = get_train_logger()

checkpoint_logger = get_checkpoints_eval_logger()


class SymbolsMelLoader(Dataset):
  """
    1) loads audio,text pairs
    2) normalizes text and converts them to sequences of one-hot vectors
    3) computes mel-spectrograms from audio files.
  """

  def __init__(self, prepare_ds_ms_data: PreparedDataList, hparams):
    data = prepare_ds_ms_data

    random.seed(hparams.seed)
    random.shuffle(data)
    self.use_saved_mels: bool = hparams.use_saved_mels
    if not hparams.use_saved_mels:
      self.mel_parser = TacotronSTFT.fromhparams(hparams)

    debug_logger.info("Reading files...")
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
      debug_logger.info("Loading mels into memory...")
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


def prepare_valloader(hparams, collate_fn: SymbolsMelCollate, valset: PreparedDataList) -> DataLoader:
  debug_logger.info(
    f"Duration valset {valset.get_total_duration_s() / 60:.2f}min / {valset.get_total_duration_s() / 60 / 60:.2f}h")

  val = SymbolsMelLoader(valset, hparams)
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


def prepare_trainloader(hparams, collate_fn: SymbolsMelCollate, trainset: PreparedDataList) -> DataLoader:
  # Get data, data loaders and collate function ready
  debug_logger.info(
    f"Duration trainset {trainset.get_total_duration_s() / 60:.2f}min / {trainset.get_total_duration_s() / 60 / 60:.2f}h")

  trn = SymbolsMelLoader(trainset, hparams)

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


def validate(model: Tacotron2, criterion: nn.Module, val_loader: DataLoader, iteration: int, logger: Tacotron2Logger):

  debug_logger.debug("Validating...")
  avg_val_loss, res = validate_core(model, criterion, val_loader)
  debug_logger.info(f"Validation loss {iteration}: {avg_val_loss:9f}")

  debug_logger.debug("Logging to tensorboard...")
  log_only_last_validation_batch = True
  if log_only_last_validation_batch:
    logger.log_validation(*res[-1], iteration)
  else:
    for entry in tqdm(res):
      logger.log_validation(*entry, iteration)
  debug_logger.debug("Finished.")

  return avg_val_loss


def train(warm_start_states: str, hparams, logdir: str, symbol_ids: SymbolIdDict, speakers: SpeakersDict, accent_ids: AccentsDict, trainset: PreparedDataList, valset: PreparedDataList, save_checkpoint_dir: str, pretrained_weights: Optional[torch.Tensor]):
  debug_logger.info("Starting new training...")
  _train(
    hparams=hparams,
    logdir=logdir,
    trainset=trainset,
    valset=valset,
    save_checkpoint_dir=save_checkpoint_dir,
    speakers=speakers,
    accents=accent_ids,
    symbols=symbol_ids,
    pretrained_weights=pretrained_weights,
    warm_start_states=warm_start_states,
    checkpoint=None,
  )


def continue_train(custom_hparams: str, logdir: str, trainset: PreparedDataList, valset: PreparedDataList, save_checkpoint_dir: str):
  debug_logger.info("Continuing training...")
  last_checkpoint_path, _ = get_last_checkpoint(save_checkpoint_dir)

  checkpoint = load_checkpoint(last_checkpoint_path, debug_logger)
  hparams = checkpoint.hparams
  symbol_ids = checkpoint.symbols
  accent_ids = checkpoint.accents
  speakers = checkpoint.speakers
  # todo apply custom hparams!
  # assert hparams.batch_size == custom.batch_size

  _train(
    hparams=hparams,
    logdir=logdir,
    trainset=trainset,
    valset=valset,
    save_checkpoint_dir=save_checkpoint_dir,
    speakers=speakers,
    accents=accent_ids,
    symbols=symbol_ids,
    pretrained_weights=None,
    warm_start_states=None,
    checkpoint=checkpoint,
  )


def _train(hparams, logdir: str, trainset: PreparedDataList, valset: PreparedDataList, save_checkpoint_dir: str, speakers: SpeakersDict, accents: AccentsDict, symbols: SymbolIdDict, checkpoint: Optional[Checkpoint], warm_start_states: Optional[dict], pretrained_weights: Optional[torch.Tensor]):
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

  debug_logger.info('Final parsed hparams:')
  debug_logger.info('\n'.join(str(hparams.values()).split(',')))

  torch.manual_seed(hparams.seed)
  torch.cuda.manual_seed(hparams.seed)
  torch.backends.cudnn.enabled = hparams.cudnn_enabled
  torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

  if checkpoint is not None:
    model, optimizer, learning_rate, iteration = model_and_optimizer_from_checkpoint(
      checkpoint, hparams)
  else:
    model, optimizer, learning_rate, iteration = model_and_optimizer_fresh(hparams)

    if warm_start_states is not None:
      warm_start_model(model, warm_start_states, hparams.ignore_layers)

    if pretrained_weights is not None:
      update_weights(model.embedding, pretrained_weights)

  debug_logger.debug("Modelweights:")
  debug_logger.debug(f"is cuda: {model.embedding.weight.is_cuda}")
  debug_logger.debug(str(model.state_dict()[SYMBOL_EMBEDDINGS_LAYER_NAME]))

  collate_fn = SymbolsMelCollate(
    n_frames_per_step=hparams.n_frames_per_step,
    padding_symbol_id=symbols.get_id(PADDING_SYMBOL),
    padding_accent_id=accents.get_id(PADDING_ACCENT)
  )

  val_loader = prepare_valloader(hparams, collate_fn, valset)
  train_loader = prepare_trainloader(hparams, collate_fn, trainset)

  batch_iterations = len(train_loader)
  if batch_iterations == 0:
    debug_logger.error("Not enough trainingdata.")
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
  logger = Tacotron2Logger(logdir)
  batch_durations: List[float] = []

  train_start = time.perf_counter()
  start = train_start
  model.train()
  continue_epoch = get_continue_epoch(iteration, batch_iterations)
  for epoch in range(continue_epoch, hparams.epochs):
    next_batch_iteration = get_continue_batch_iteration(iteration, batch_iterations)
    for batch_iteration, batch in enumerate(train_loader):
      need_to_skip_batch = skip_batch(
        batch_iteration=batch_iteration,
        continue_batch_iteration=next_batch_iteration
      )
      if need_to_skip_batch:
        debug_logger.debug(f"Skipped batch {batch_iteration + 1}.")
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
      debug_logger.info(" | ".join([
        f"Epoch: {get_formatted_current_total(epoch + 1, hparams.epochs)}",
        f"Iteration: {get_formatted_current_total(batch_iteration + 1, batch_iterations)}",
        f"Total iteration: {get_formatted_current_total(iteration, hparams.epochs * batch_iterations)}",
        f"Train loss: {reduced_loss:.6f}",
        f"Grad Norm: {grad_norm:.6f}",
        f"Duration: {duration:.2f}s/it",
        f"Avg. duration: {np.mean(batch_durations):.2f}s/it",
        f"Total Duration: {(time.perf_counter() - train_start) / 60 / 60:.2f}h"
      ]))

      logger.log_training(reduced_loss, grad_norm, learning_rate,
                          duration, iteration)

      save_it = check_save_it(epoch, iteration, save_it_settings)
      if save_it:
        checkpoint = Checkpoint(
          state_dict=model.state_dict(),
          optimizer=optimizer.state_dict(),
          learning_rate=learning_rate,
          iteration=iteration,
          hparams=hparams,
          symbols=symbols,
          accents=accents,
          speakers=speakers
        )

        checkpoint_path = os.path.join(
          save_checkpoint_dir, get_pytorch_filename(iteration))
        save_checkpoint(checkpoint_path, checkpoint)

        valloss = validate(model, criterion, val_loader, iteration, logger)
        # if rank == 0:
        log_checkpoint_score(iteration, grad_norm,
                             reduced_loss, valloss, epoch, batch_iteration)

  duration_s = time.time() - complete_start
  debug_logger.info(f'Finished training. Total duration: {duration_s / 60:.2f}min')


def update_symbol_embeddings(model: Tacotron2, weights: torch.Tensor):
  debug_logger.info("Loading pretrained mapped embeddings...")
  update_weights(model.embedding, weights)


def model_and_optimizer_fresh(hparams):
  debug_logger.info("Starting new model...")
  model = load_model(hparams, None, debug_logger)

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


def model_and_optimizer_from_checkpoint(checkpoint: Checkpoint, updated_hparams):
  debug_logger.info("Continuing training from checkpoint...")
  learning_rate: float = updated_hparams.learning_rate
  if updated_hparams.use_saved_learning_rate:
    learning_rate = checkpoint.learning_rate

  model = load_model(updated_hparams, checkpoint.state_dict, debug_logger)

  # warn: use saved learning rate is ignored here
  optimizer = torch.optim.Adam(
    params=model.parameters(),
    lr=learning_rate,
    weight_decay=updated_hparams.weight_decay
  )
  optimizer.load_state_dict(checkpoint.optimizer)

  return model, optimizer, learning_rate, checkpoint.iteration


def save_checkpoint(checkpoint_path: str, checkpoint: Checkpoint):
  debug_logger.info(f"Saving model at iteration {checkpoint.iteration}...")
  checkpoint_dict = asdict(checkpoint)
  torch.save(checkpoint_dict, checkpoint_path)
  debug_logger.info(f"Saved model to '{checkpoint_path}'.")


def load_checkpoint(checkpoint_path: str, logger: Logger) -> Checkpoint:
  assert os.path.isfile(checkpoint_path)
  logger.info(f"Loading tacotron model '{checkpoint_path}'...")
  checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
  checkpoint = Checkpoint(**checkpoint_dict)
  logger.info(f"Loaded model at iteration {checkpoint.iteration}.")
  logger.info(f'Including {len(checkpoint.symbols)} symbols.')
  logger.info(f'Including {len(checkpoint.accents)} accents.')
  logger.info(f'Including {len(checkpoint.speakers)} speaker(s).')
  return checkpoint


def warm_start_model(model: Tacotron2, warm_start_states: dict, ignore_layers: List[str]):
  debug_logger.info("Loading states from pretrained model...")
  # The default value from HParams is [""], an empty list was not working.
  ignore_layers.extend([
    SYMBOL_EMBEDDINGS_LAYER_NAME,
    SPEAKER_EMBEDDINGS_LAYER_NAME
  ])

  model_dict = {k: v for k, v in warm_start_states.items() if k not in ignore_layers}
  _update_model_state_dict(model, model_dict)


def _update_model_state_dict(model: Tacotron2, updates: dict):
  dummy_dict = model.state_dict()
  dummy_dict.update(updates)
  model.load_state_dict(dummy_dict)


def log_checkpoint_score(iteration: int, gradloss: float, trainloss: float, valloss: float, epoch: int, i: int):
  loss_avg = (trainloss + valloss) / 2
  msg = f"{iteration}\tepoch-{epoch}\tit-{i}\tgradloss-{gradloss:.6f}\ttrainloss-{trainloss:.6f}\tvalidationloss-{valloss:.6f}\tavg-train-val-{loss_avg:.6f}"
  checkpoint_logger.info(msg)


def load_state_dict_from(model_path: str) -> dict:
  warm_start_states = load_checkpoint(model_path, debug_logger).state_dict
  return warm_start_states


def load_symbol_embedding_weights_from(model_path: str) -> torch.Tensor:
  model_state_dict = load_checkpoint(model_path, debug_logger).state_dict
  pretrained_weights = model_state_dict[SYMBOL_EMBEDDINGS_LAYER_NAME]
  return pretrained_weights


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


def map_weights(model_symbols_id_map: OrderedDictType[int, int], model_weights, trained_weights):
  for map_to_model_symbol_id, map_from_symbol_id in model_symbols_id_map.items():
    assert 0 <= map_to_model_symbol_id < model_weights.shape[0]
    assert 0 <= map_from_symbol_id < trained_weights.shape[0]

    debug_logger.debug(f"Mapped {map_from_symbol_id} to {map_to_model_symbol_id}.")
    model_weights[map_to_model_symbol_id] = trained_weights[map_from_symbol_id]


def get_mapped_symbol_weights(model_symbols: SymbolIdDict, trained_weights: torch.Tensor, trained_symbols: SymbolIdDict, custom_mapping: Optional[SymbolsMap], hparams) -> torch.Tensor:
  symbols_match_not_model = trained_weights.shape[0] != len(trained_symbols)
  if symbols_match_not_model:
    debug_logger.exception(
      f"Weights mapping: symbol space from pretrained model ({trained_weights.shape[0]}) did not match amount of symbols ({len(trained_symbols)}).")
    raise Exception()

  symbols_map = get_map(
    dest_symbols=model_symbols.get_all_symbols(),
    orig_symbols=trained_symbols.get_all_symbols(),
    symbols_mapping=custom_mapping,
    logger=debug_logger
  )

  symbols_id_map = symbols_map_to_symbols_ids_map(
    dest_symbols=model_symbols,
    orig_symbols=trained_symbols,
    symbols_mapping=symbols_map,
    logger=debug_logger
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
    trained_weights=trained_weights
  )

  symbols_wo_mapping = symbols_map.get_symbols_without_mapping()
  not_existing_symbols = model_symbols.get_all_symbols() - symbols_map.keys()
  no_mapping = symbols_wo_mapping | not_existing_symbols
  if len(no_mapping) > 0:
    debug_logger.warning(f"Following symbols were not mapped: {no_mapping}")
  else:
    debug_logger.info("All symbols were mapped.")

  return model_weights


def eval_checkpoints(custom_hparams: Optional[str], checkpoint_dir: str, select: int, min_it: int, max_it: int, n_symbols: int, n_accents: int, n_speakers: int, valset: PreparedDataList):
  its = get_all_checkpoint_iterations(checkpoint_dir)
  debug_logger.info(f"Available iterations {its}")
  filtered_its = filter_checkpoints(its, select, min_it, max_it)
  if len(filtered_its) > 0:
    debug_logger.info(f"Selected iterations: {filtered_its}")
  else:
    debug_logger.info("None selected. Exiting.")
    return

  hparams = create_hparams(n_speakers, n_symbols, n_accents, custom_hparams)

  collate_fn = SymbolsMelCollate(
    hparams.n_frames_per_step,
    padding_symbol_id=0,  # TODO: refactor
    padding_accent_id=0  # TODO: refactor
  )
  val_loader = prepare_valloader(hparams, collate_fn, valset)

  result = []
  for checkpoint_iteration in tqdm(filtered_its):
    criterion = Tacotron2Loss()
    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)
    full_checkpoint_path, _ = get_custom_checkpoint(checkpoint_dir, checkpoint_iteration)
    checkpoint_loaded = load_checkpoint(full_checkpoint_path, debug_logger)
    model = load_model(hparams, checkpoint_loaded.state_dict, debug_logger)
    val_loss, _ = validate_core(model, criterion, val_loader)
    result.append((checkpoint_iteration, val_loss))
    debug_logger.info(f"Validation loss {checkpoint_iteration}: {val_loss:9f}")

  debug_logger.info("Result...")
  debug_logger.info("Sorted after checkpoints:")

  result.sort()
  for cp, loss in result:
    debug_logger.info(f"Validation loss {cp}: {loss:9f}")

  result = [(b, a) for a, b in result]
  result.sort()

  debug_logger.info("Sorted after scores:")
  for loss, cp in result:
    debug_logger.info(f"Validation loss {cp}: {loss:9f}")
