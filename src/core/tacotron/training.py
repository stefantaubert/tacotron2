import logging
import os
import random
import time
from math import sqrt
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.core.common.accents_dict import PADDING_ACCENT, AccentsDict
from src.core.common.symbol_id_dict import PADDING_SYMBOL, SymbolIdDict
from src.core.common.symbols_map import SymbolsMap, get_symbols_id_mapping
from src.core.common.taco_stft import TacotronSTFT
from src.core.common.text import deserialize_list
from src.core.common.train import (get_all_checkpoint_iterations,
                                   get_custom_checkpoint, get_last_checkpoint)
from src.core.pre.merge_ds import PreparedData, PreparedDataList
from src.core.tacotron.hparams import create_hparams
from src.core.tacotron.logger import Tacotron2Logger
from src.core.tacotron.model import (Tacotron2, get_model_symbol_id,
                                     speaker_embeddings_layer_name,
                                     symbol_embeddings_layer_name)


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
      assert len(symbol_ids) == len(accent_ids)

      model_symbol_ids = []
      symbol_accent_ids = zip(symbol_ids, accent_ids)
      for symbol_id, accent_id in symbol_accent_ids:
        model_symbol_id = get_model_symbol_id(
          symbol_id, accent_id, hparams.n_symbols, hparams.accents_use_own_symbols)
        model_symbol_ids.append(model_symbol_id)

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

  def __getitem__(self, index) -> Tuple[torch.IntTensor, torch.IntTensor, torch.Tensor, int]:
    # return self.cache[index]
    #debug_logger.debug(f"getitem called {index}")
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
    #debug_logger.debug(f"getitem finished {index}")
    return symbols_tensor_cloned, accents_tensor_cloned, mel_tensor, speaker_id

  def __len__(self):
    return len(self.data)


class SymbolsMelCollate():
  """ Zero-pads model inputs and targets based on number of frames per step
  """

  def __init__(self, n_frames_per_step):
    self.n_frames_per_step = n_frames_per_step

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
    symbol_padding_id = 0 # "_"
    torch.nn.init.constant_(symbols_padded, symbol_padding_id)

    accents_padded = torch.LongTensor(len(batch), max_input_len)
    accent_padding_id = 0 # "none"
    torch.nn.init.constant_(accents_padded, accent_padding_id)

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
    #len_x = []
    speaker_ids = []
    for i, batch_id in enumerate(ids_sorted_decreasing):
      #len_symb = batch[batch_id][0].get_shape()[0]
      # len_x.append(len_symb)
      _, _, _, speaker_id = batch[batch_id]
      speaker_ids.append(speaker_id)

    #len_x = torch.Tensor(len_x)
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
  def forward(self, model_output, targets):
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


def prepare_valloader(hparams, collate_fn: SymbolsMelCollate, valset: PreparedDataList):
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


def prepare_trainloader(hparams, collate_fn: SymbolsMelCollate, trainset: PreparedDataList):
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


def load_model(hparams, logger: logging.Logger):
  model = Tacotron2(hparams, logger).cuda()
  # if hparams.fp16_run:
  #   model.decoder.attention_layer.score_mask_value = finfo('float16').min

  # if hparams.distributed_run:
  #   model = apply_gradient_allreduce(model)

  return model


def validate(model: Tacotron2, criterion: nn.Module, val_loader: DataLoader, iteration: int, logger: Optional[Tacotron2Logger] = None):
  """Handles all the validation scoring and printing"""
  debug_logger.debug("Validating...")
  model.eval()
  with torch.no_grad():
    val_loss = 0.0
    for batch in tqdm(val_loader):
      x, y = Tacotron2.parse_batch(batch)
      y_pred = model(x)
      loss = criterion(y_pred, y)
      # if distributed_run:
      #   reduced_val_loss = reduce_tensor(loss.data, n_gpus).item()
      # else:
      #  reduced_val_loss = loss.item()
      reduced_val_loss = loss.item()
      val_loss += reduced_val_loss
    val_loss = val_loss / len(val_loader)
  model.train()

  debug_logger.info(f"Validation loss {iteration}: {val_loss:9f}")
  if logger:
    logger.log_validation(val_loss, model, y, y_pred, iteration)

  return val_loss


def train_core(hparams, logdir: str, trainset: PreparedDataList, valset: PreparedDataList, save_checkpoint_dir: str, iteration: int, model: Tacotron2, optimizer, learning_rate: float):
  complete_start = time.time()
  logger = Tacotron2Logger(logdir)

  debug_logger.info('Final parsed hparams:')
  debug_logger.info('\n'.join(str(hparams.values()).split(',')))

  collate_fn = SymbolsMelCollate(n_frames_per_step=hparams.n_frames_per_step)

  val_loader = prepare_valloader(hparams, collate_fn, valset)
  train_loader = prepare_trainloader(hparams, collate_fn, trainset)

  if len(train_loader) == 0:
    debug_logger.error("Not enough trainingdata.")
    return False

  criterion = Tacotron2Loss()

  debug_logger.debug("Modelweights:")
  debug_logger.debug(str(model.state_dict()[symbol_embeddings_layer_name]))

  model.train()
  # is_overflow = False
  total_its = hparams.epochs * len(train_loader)
  # ================ MAIN TRAINING LOOP! ===================
  train_start = time.perf_counter()
  epoch_offset = max(0, int(iteration / len(train_loader)))
  batch_durations: List[float] = []
  for epoch in range(epoch_offset, hparams.epochs):
    debug_logger.info(f"Epoch: {epoch}")
    start = time.perf_counter()
    for i, batch in enumerate(train_loader):
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

      # if not is_overflow and rank == 0:
      duration = time.perf_counter() - start
      batch_durations.append(duration)
      debug_logger.info(" | ".join([
        f"Epoch: {str(epoch).zfill(len(str(hparams.epochs)))}/{hparams.epochs}",
        f"Iteration: {str(i).zfill(len(str(len(train_loader) - 1)))}/{len(train_loader) - 1}",
        f"Total iteration: {str(iteration).zfill(len(str(total_its)))}/{total_its}",
        f"Train loss: {reduced_loss:.6f}",
        f"Grad Norm: {grad_norm:.6f}",
        f"Duration: {duration:.2f}s/it",
        f"Avg. duration: {np.mean(batch_durations):.2f}s/it",
        f"Total Duration: {(time.perf_counter() - train_start) / 60 / 60:.2f}h"
      ]))
      logger.log_training(reduced_loss, grad_norm, learning_rate, duration, iteration)

      # is_last_it = i + 1 == len(train_loader)
      # is_last_epoch = epoch + 1 == hparams.epochs
      # save_epoch = is_last_it and is_last_epoch and
      # is_first_iteration = iteration == 0
      # is_checkpoint_iteration = (hparams.iters_per_checkpoint > 0 and (iteration % hparams.iters_per_checkpoint == 0)) or iteration == 0

      #save_iteration = is_checkpoint_iteration or is_last_it or is_last_epoch
      # if not is_overflow and (iteration % hparams.iters_per_checkpoint == 0):
      save_iteration = (hparams.iters_per_checkpoint > 0 and (iteration %
                                                              hparams.iters_per_checkpoint == 0)) or iteration == 0
      if save_iteration:
        save_checkpoint(model, optimizer, learning_rate, iteration, save_checkpoint_dir)
        valloss = validate(model, criterion, val_loader, iteration, logger)
        # if rank == 0:
        log_checkpoint_score(iteration, grad_norm, reduced_loss, valloss, epoch, i)

      iteration += 1
      start = time.perf_counter()

    checkpoint_was_already_created = save_iteration
    save_epoch = not checkpoint_was_already_created and hparams.epochs_per_checkpoint > 0 and (
      epoch % hparams.epochs_per_checkpoint == 0)
    if save_epoch:
      save_checkpoint(model, optimizer, learning_rate, iteration - 1, save_checkpoint_dir)
      valloss = validate(model, criterion, val_loader, iteration - 1, logger)
      # if rank == 0:
      log_checkpoint_score(iteration - 1, grad_norm, reduced_loss, valloss, epoch, i)

  checkpoint_was_already_created = save_iteration or save_epoch
  if not checkpoint_was_already_created:
    save_checkpoint(model, optimizer, learning_rate, iteration - 1, save_checkpoint_dir)
    valloss = validate(model, criterion, val_loader, iteration - 1, logger)
    # if rank == 0:
    log_checkpoint_score(iteration - 1, grad_norm, reduced_loss, valloss, epoch, i)

  duration_s = time.time() - complete_start
  debug_logger.info(f'Finished training. Total duration: {duration_s / 60:.2f}min')


def continue_train(custom_hparams: str, n_symbols: int, n_speakers: int, n_accents: int, logdir: str, trainset: PreparedDataList, valset: PreparedDataList, save_checkpoint_dir: str):
  debug_logger.info("Continuing training...")
  hparams = create_hparams(n_speakers, n_symbols, n_accents, custom_hparams)

  last_checkpoint_path, _ = get_last_checkpoint(save_checkpoint_dir)

  model, optimizer, learning_rate, iteration = _train("", "", last_checkpoint_path, hparams)
  train_core(hparams, logdir, trainset, valset, save_checkpoint_dir,
             iteration, model, optimizer, learning_rate)


def train(warm_start_model_path: str, custom_hparams: str, logdir: str, symbol_ids: SymbolIdDict, n_speakers: int, accent_ids: AccentsDict, trainset: PreparedDataList, valset: PreparedDataList, save_checkpoint_dir: str, trained_weights: Optional[torch.Tensor], symbols_map: Optional[SymbolsMap], trained_symbols_conv: Optional[SymbolIdDict]):
  assert symbol_ids.get_id(PADDING_SYMBOL) == 0
  assert accent_ids.get_id(PADDING_ACCENT) == 0

  n_symbols = len(symbol_ids)
  hparams = create_hparams(n_speakers, n_symbols, accent_ids, custom_hparams)

  mapped_emb_weights = None
  if trained_weights is not None:
    # TODO: recognize accents
    mapped_emb_weights = get_mapped_embedding_weights(
      model_symbols=symbol_ids,
      trained_weights=trained_weights,
      trained_symbols=trained_symbols_conv,
      symbols_mapping=symbols_map
    )

  model, optimizer, learning_rate, iteration = _train(
    warm_start_model_path, mapped_emb_weights, "", hparams)
  train_core(hparams, logdir, trainset, valset, save_checkpoint_dir,
             iteration, model, optimizer, learning_rate)


def _train(warm_start_model_path: str, mapped_emb_weights: torch.Tensor, checkpoint_path: str, hparams):
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
  # if hparams.distributed_run:
  #   init_distributed(hparams, n_gpus, rank, group_name, training_dir_path)

  assert hparams.n_symbols
  assert hparams.n_speakers
  assert hparams.n_accents

  torch.backends.cudnn.enabled = hparams.cudnn_enabled
  torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

  torch.manual_seed(hparams.seed)
  torch.cuda.manual_seed(hparams.seed)

  model = load_model(hparams, debug_logger)
  learning_rate = hparams.learning_rate
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                               weight_decay=hparams.weight_decay)

  # if hparams.fp16_run:
  #   from apex import amp
  #   model, optimizer = amp.initialize(model, optimizer, opt_level='O2')

  # if hparams.distributed_run:
  #   model = apply_gradient_allreduce(model)

  # Load checkpoint if one exists
  iteration = 0
  if checkpoint_path:
    model, optimizer, _learning_rate, iteration = load_checkpoint(checkpoint_path, model, optimizer)

    if hparams.use_saved_learning_rate:
      learning_rate = _learning_rate
    iteration += 1  # next iteration is iteration + 1
  else:
    if warm_start_model_path:
      assert os.path.isfile(warm_start_model_path)
      # raise Exception("Warm start was not possible because the path to the model was not valid.")
      warm_start_model(warm_start_model_path, model, hparams.ignore_layers)
    else:
      debug_logger.info("Starting new model...")

    if mapped_emb_weights is not None:
      debug_logger.info("Loading pretrained, mapped embeddings...")
      init_symbol_embedding_weights(model, mapped_emb_weights)

  return model, optimizer, learning_rate, iteration


def save_checkpoint(model: Tacotron2, optimizer: torch.optim.Optimizer, learning_rate: float, iteration: int, parent_dir: str):
  filepath = os.path.join(parent_dir, get_pytorch_filename(iteration))
  debug_logger.info(f"Saving model and optimizer state at iteration {iteration} to {filepath}")
  save_checkpoint_dict(filepath, model.state_dict(),
                       optimizer.state_dict(), learning_rate, iteration)


def load_checkpoint(checkpoint_path, model: Tacotron2, optimizer: torch.optim.Optimizer) -> Tuple[Tacotron2, torch.optim.Optimizer, float, int]:
  debug_logger.info(f"Loading checkpoint '{checkpoint_path}'")
  model_dict, opt_dict, lr, it = load_checkpoint_dict(checkpoint_path)
  model.load_state_dict(model_dict)
  optimizer.load_state_dict(opt_dict)
  debug_logger.info(f"Loaded checkpoint '{checkpoint_path}' from iteration {it}")
  return model, optimizer, lr, it


def save_checkpoint_dict(checkpoint_path: str, model_state_dict: dict, optimizer_state_dict: dict, learning_rate: float, iteration: int):
  checkpoint_dict = {
    'state_dict': model_state_dict,
    'optimizer': optimizer_state_dict,
    'learning_rate': learning_rate,
    'iteration': iteration
  }
  torch.save(checkpoint_dict, checkpoint_path)


def load_checkpoint_dict(checkpoint_path: str) -> Tuple[dict, dict, float, int]:
  assert os.path.isfile(checkpoint_path)
  checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
  return (
    checkpoint_dict['state_dict'],
    checkpoint_dict['optimizer'],
    checkpoint_dict['learning_rate'],
    checkpoint_dict['iteration']
  )


def warm_start_model(checkpoint_path: str, model: str, ignore_layers: List[str]):
  model_state_dict = load_checkpoint_dict(checkpoint_path)[0]
  # The default value from HParams is [""], an empty list was not working.
  ignore_layers.extend([
    symbol_embeddings_layer_name,
    speaker_embeddings_layer_name
  ])
  model_dict = {k: v for k, v in model_state_dict.items() if k not in ignore_layers}
  _update_model_state_dict(model, model_dict)


def init_symbol_embedding_weights(model: Tacotron2, emb_weights: torch.Tensor):
  update = {
    symbol_embeddings_layer_name: emb_weights
  }
  _update_model_state_dict(model, update)


def _update_model_state_dict(model: Tacotron2, updates: dict):
  dummy_dict = model.state_dict()
  dummy_dict.update(updates)
  model.load_state_dict(dummy_dict)


def log_checkpoint_score(iteration: int, gradloss: float, trainloss: float, valloss: float, epoch: int, i: int):
  loss_avg = (trainloss + valloss) / 2
  msg = f"{iteration}\tepoch-{epoch}\tit-{i}\tgradloss-{gradloss:.6f}\ttrainloss-{trainloss:.6f}\tvalidationloss-{valloss:.6f}\tavg-train-val-{loss_avg:.6f}"
  checkpoint_logger.info(msg)


def get_uniform_weights(n_symbols: int, emb_dim: int) -> torch.Tensor:
  weight = torch.zeros(size=(n_symbols, emb_dim))
  std = sqrt(2.0 / (n_symbols + emb_dim))
  val = sqrt(3.0) * std  # uniform bounds for std
  nn.init.uniform_(weight, -val, val)
  return weight


def load_symbol_embedding_weights_from(model_path: str) -> torch.Tensor:
  model_state_dict = load_checkpoint_dict(model_path)[0]
  pretrained_weights = model_state_dict[symbol_embeddings_layer_name]
  return pretrained_weights

# def load_weights(weights_path: str):
#   assert os.path.isfile(weights_path)
#   weights = np.load(weights_path)
#   weights = torch.from_numpy(weights)
#   return weights


def get_mapped_embedding_weights(model_symbols: SymbolIdDict, trained_weights: torch.Tensor, trained_symbols: SymbolIdDict, symbols_mapping: Optional[SymbolsMap] = None) -> torch.Tensor:
  model_weights = get_uniform_weights(len(model_symbols), trained_weights.shape[1])
  return get_mapped_embedding_weights_core(model_weights, model_symbols, trained_weights, trained_symbols, symbols_mapping)


def get_mapped_embedding_weights_core(model_weights: torch.Tensor, model_symbols: SymbolIdDict, trained_weights: torch.Tensor, trained_symbols: SymbolIdDict, symbols_mapping: Optional[SymbolsMap] = None) -> torch.Tensor:
  assert model_weights.shape[0] == len(model_symbols)

  symbols_match_not_model = trained_weights.shape[0] != len(trained_symbols)
  if symbols_match_not_model:
    debug_logger.exception(
      f"Weights mapping: symbol space from pretrained model ({trained_weights.shape[0]}) did not match amount of symbols ({len(trained_symbols)}).")
    raise Exception()

  mapping = get_symbols_id_mapping(model_symbols, trained_symbols, symbols_mapping, debug_logger)
  for map_to_symbol_id, map_from_symbol_id in mapping.items():
    assert 0 <= map_to_symbol_id < model_weights.shape[0]
    assert 0 <= map_from_symbol_id < trained_weights.shape[0]

    model_weights[map_to_symbol_id] = trained_weights[map_from_symbol_id]

  return model_weights


def _filter_checkpoints(iterations: List[int], select: int, min_it: int, max_it: int):
  if not select:
    select = 0
  if not min_it:
    min_it = 0
  if not max_it:
    max_it = max(iterations)
  process_checkpoints = [checkpoint for checkpoint in iterations if checkpoint %
                         select == 0 and checkpoint >= min_it and checkpoint <= max_it]

  return process_checkpoints


def eval_checkpoints(custom_hparams: Optional[str], checkpoint_dir: str, select: int, min_it: int, max_it: int, n_symbols: int, n_accents: int, n_speakers: int, valset: PreparedDataList):
  its = get_all_checkpoint_iterations(checkpoint_dir)
  debug_logger.info(f"Available iterations {its}")
  filtered_its = _filter_checkpoints(its, select, min_it, max_it)
  if len(filtered_its):
    debug_logger.info(f"Selected iterations: {filtered_its}")
  else:
    debug_logger.info("None selected. Exiting.")
    return

  hparams = create_hparams(n_speakers, n_symbols, n_accents, custom_hparams)

  collate_fn = SymbolsMelCollate(hparams.n_frames_per_step)
  val_loader = prepare_valloader(hparams, collate_fn, valset)

  result = []
  for checkpoint in tqdm(filtered_its):
    criterion = Tacotron2Loss()
    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)
    model = load_model(hparams, debug_logger)
    learning_rate = hparams.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=hparams.weight_decay)

    full_checkpoint_path, _ = get_custom_checkpoint(checkpoint_dir, checkpoint)

    model, _, _, _ = load_checkpoint(full_checkpoint_path, model, optimizer)
    val_loss = validate(model, criterion, val_loader, checkpoint)
    result.append((checkpoint, val_loss))
    debug_logger.info(f"Validation loss {checkpoint}: {val_loss:9f}")

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
