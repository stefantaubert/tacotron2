import math
import os
import time
from functools import partial

import numpy as np
import torch
from numpy import finfo
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.core.common import log, reset_log, get_last_checkpoint
from src.core.pre import PreparedData, PreparedDataList, SpeakersIdDict
from torch import nn
from src.core.tacotron.hparams import create_hparams
from src.core.tacotron.logger import Tacotron2Logger
from src.core.tacotron.model import Tacotron2
import random
import logging

from src.core.pre import SymbolConverter

def get_train_logger():
  return logging.getLogger("taco-train")

# def get_checkpoints_eval_logger():
#   return logging.getLogger("taco-train-checkpoints")

debug_logger = get_train_logger()

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
    
    debug_logger.info("Reading mels...")
    self.data = {}
    values: PreparedData
    for i, values in enumerate(tqdm(data)):
      symbol_ids = SymbolConverter.deserialize_symbol_ids(values.serialized_updated_ids)
      symbols_tensor = torch.IntTensor(symbol_ids)
      self.data[i] = (symbols_tensor, values.mel_path, values.speaker_id)
    
    if hparams.cache_mels:
      debug_logger.info("Loading mels into memory...")
      self.cache = {}
      vals: tuple
      for i, vals in tqdm(self.data.items()):
        mel_tensor = torch.load(vals[1], map_location='cpu')
        self.cache[i] = mel_tensor
    self.use_cache = hparams.cache_mels

  def __getitem__(self, index):
    #return self.cache[index]
    symbols_tensor, mel_path, speaker_id = self.data[index]
    if self.use_cache:
      mel_tensor = self.cache[index].clone().detach()
    else:
      mel_tensor = torch.load(mel_path, map_location='cpu')
    return symbols_tensor.clone().detach(), mel_tensor, speaker_id

  def __len__(self):
    return len(self.data)


class SymbolsMelCollate():
  """ Zero-pads model inputs and targets based on number of frames per step
  """
  def __init__(self, n_frames_per_step):
    self.n_frames_per_step = n_frames_per_step

  def __call__(self, batch):
    """Collate's training batch from normalized text and mel-spectrogram
    PARAMS
    ------
    batch: [text_normalized, mel_normalized]
    """
    # Right zero-pad all one-hot text sequences to max input length
    input_lengths, ids_sorted_decreasing = torch.sort(torch.LongTensor([len(x[0]) for x in batch]), dim=0, descending=True)
    max_input_len = input_lengths[0]

    text_padded = torch.LongTensor(len(batch), max_input_len)
    text_padded.zero_()
    for i in range(len(ids_sorted_decreasing)):
      text = batch[ids_sorted_decreasing[i]][0]
      text_padded[i, :text.size(0)] = text

    # Right zero-pad mel-spec
    num_mels = batch[0][1].size(0)
    max_target_len = max([x[1].size(1) for x in batch])
    if max_target_len % self.n_frames_per_step != 0:
      max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
      assert max_target_len % self.n_frames_per_step == 0

    # include mel padded and gate padded
    mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
    mel_padded.zero_()
    gate_padded = torch.FloatTensor(len(batch), max_target_len)
    gate_padded.zero_()
    output_lengths = torch.LongTensor(len(batch))
    for i in range(len(ids_sorted_decreasing)):
      mel = batch[ids_sorted_decreasing[i]][1]
      mel_padded[i, :, :mel.size(1)] = mel
      gate_padded[i, mel.size(1)-1:] = 1
      output_lengths[i] = mel.size(1)

    # count number of items - characters in text
    #len_x = []
    speaker_ids = []
    for i in range(len(ids_sorted_decreasing)):
      #len_symb = batch[ids_sorted_decreasing[i]][0].get_shape()[0]
      #len_x.append(len_symb)
      speaker_ids.append(batch[ids_sorted_decreasing[i]][2])

    #len_x = torch.Tensor(len_x)
    speaker_ids = torch.Tensor(speaker_ids)

    return text_padded, input_lengths, mel_padded, gate_padded, output_lengths, speaker_ids

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
  def __init__(self):
    super(Tacotron2Loss, self).__init__()

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

def prepare_dataloaders(hparams, trainset: PreparedDataList, valset: PreparedDataList):
  # Get data, data loaders and collate function ready
  trainset = SymbolsMelLoader(trainset, hparams)
  valset = SymbolsMelLoader(valset, hparams)

  collate_fn = SymbolsMelCollate(hparams.n_frames_per_step)

  train_sampler = None
  shuffle = True

  # if hparams.distributed_run:
  #   train_sampler = DistributedSampler(trainset)
  #   shuffle = False

  train_loader = DataLoader(
    trainset,
    num_workers=1,
    shuffle=shuffle,
    sampler=train_sampler,
    batch_size=hparams.batch_size,
    pin_memory=False,
    drop_last=True,
    collate_fn=collate_fn
  )
  
  return train_loader, valset, collate_fn

def load_model(hparams):
  model = Tacotron2(hparams).cuda()
  # if hparams.fp16_run:
  #   model.decoder.attention_layer.score_mask_value = finfo('float16').min

  # if hparams.distributed_run:
  #   model = apply_gradient_allreduce(model)

  return model


def warm_start_model(checkpoint_path, model, ignore_layers):
  assert os.path.isfile(checkpoint_path)
  checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
  model_dict = checkpoint_dict['state_dict']
  if len(ignore_layers) > 0:
    model_dict = {k: v for k, v in model_dict.items() if k not in ignore_layers}
    dummy_dict = model.state_dict()
    dummy_dict.update(model_dict)
    model_dict = dummy_dict
  model.load_state_dict(model_dict)

def validate_core(model, criterion, valset, batch_size, collate_fn):
  """Handles all the validation scoring and printing"""
  model.eval()
  with torch.no_grad():
    val_sampler = None

    # if distributed_run:
    #   val_sampler = DistributedSampler(valset)

    val_loader = DataLoader(valset, sampler=val_sampler, num_workers=1,
                shuffle=False, batch_size=batch_size,
                pin_memory=False, collate_fn=collate_fn)

    val_loss = 0.0
    debug_logger.debug("Validating...")
    for i, batch in enumerate(tqdm(val_loader)):
      x, y = model.parse_batch(batch)
      y_pred = model(x)
      loss = criterion(y_pred, y)
      # if distributed_run:
      #   reduced_val_loss = reduce_tensor(loss.data, n_gpus).item()
      # else:
      #  reduced_val_loss = loss.item()
      reduced_val_loss = loss.item()
      val_loss += reduced_val_loss
    val_loss = val_loss / (i + 1)

  model.train()
  return val_loss, model, y, y_pred

def validate(model, criterion, valset, iteration, batch_size, collate_fn, logger: Tacotron2Logger):
  val_loss, model, y, y_pred = validate_core(model, criterion, valset, batch_size, collate_fn)

  #if rank == 0:
  debug_logger.info("Validation loss {}: {:9f}".format(iteration, val_loss))
  logger.log_validation(val_loss, model, y, y_pred, iteration)
  
  return val_loss

def train_core(hparams, logger: Tacotron2Logger, trainset: PreparedDataList, valset: PreparedDataList, save_checkpoint_dir: str, save_checkpoint_log_dir: str, iteration: int, model, optimizer, learning_rate):
  complete_start = time.time()

  debug_logger.info('Final parsed hparams:')
  x = '\n'.join(str(hparams.values()).split(','))
  debug_logger.info(x)

  debug_logger.info("Epochs: {}".format(hparams.epochs))
  debug_logger.info("Batchsize: {}".format(hparams.batch_size))
  debug_logger.info("FP16 Run: {}".format(hparams.fp16_run))
  debug_logger.info("Dynamic Loss Scaling: {}".format(hparams.dynamic_loss_scaling))
  debug_logger.info("Distributed Run: {}".format(hparams.distributed_run))
  debug_logger.info("cuDNN Enabled: {}".format(hparams.cudnn_enabled))
  debug_logger.info("cuDNN Benchmark: {}".format(hparams.cudnn_benchmark))

  criterion = Tacotron2Loss()

  train_loader, valset, collate_fn = prepare_dataloaders(hparams, trainset, valset)

  if not len(train_loader):
    debug_logger.error("Not enough trainingdata.")
    return False

  debug_logger.info("Modelweights:")
  debug_logger.info(str(model.state_dict()['embedding.weight']))
  debug_logger.info("Iterations per epoch: {}".format(len(train_loader)))

  model.train()
  # is_overflow = False
  total_its = hparams.epochs * len(train_loader)
  # ================ MAIN TRAINING LOOP! ===================
  train_start = time.perf_counter()
  epoch_offset = max(0, int(iteration / len(train_loader)))
  for epoch in range(epoch_offset, hparams.epochs):
    debug_logger.info("Epoch: {}".format(epoch))
    
    for i, batch in enumerate(train_loader):
      start = time.perf_counter()
      for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

      model.zero_grad()
      x, y = model.parse_batch(batch)
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

      #if not is_overflow and rank == 0:
      duration = time.perf_counter() - start
      debug_logger.info("Epoch: {}/{} | Iteration: {}/{} | Total iteration: {}/{} | Train loss: {:.6f} | Grad Norm: {:.6f} | Duration: {:.2f}s/it | Total Duration: {:.2f}h".format(
        str(epoch).zfill(len(str(hparams.epochs))),
        hparams.epochs,
        str(i).zfill(len(str(len(train_loader) - 1))),
        len(train_loader) - 1,
        str(iteration).zfill(len(str(total_its))),
        total_its,
        reduced_loss,
        grad_norm,
        duration,
        (time.perf_counter() - train_start) / 60 / 60
      ))
      logger.log_training(reduced_loss, grad_norm, learning_rate, duration, iteration)

      #if not is_overflow and (iteration % hparams.iters_per_checkpoint == 0):
      save_iteration = (hparams.iters_per_checkpoint > 0 and (iteration % hparams.iters_per_checkpoint == 0)) or iteration == 0
      if save_iteration:
        valloss = validate(model, criterion, valset, iteration, hparams.batch_size, collate_fn, logger)
        #if rank == 0:
        save_checkpoint(model, optimizer, learning_rate, iteration, save_checkpoint_dir)
        save_checkpoint_score(save_checkpoint_log_dir, iteration, grad_norm, reduced_loss, valloss, epoch, i)
      iteration += 1

    checkpoint_was_already_created = save_iteration
    save_epoch = not checkpoint_was_already_created and hparams.epochs_per_checkpoint > 0 and (epoch % hparams.epochs_per_checkpoint == 0)
    if save_epoch:
      valloss = validate(model, criterion, valset, iteration - 1, hparams.batch_size, collate_fn, logger)
      #if rank == 0:
      save_checkpoint(model, optimizer, learning_rate, iteration - 1, save_checkpoint_dir)
      save_checkpoint_score(save_checkpoint_log_dir, iteration - 1, grad_norm, reduced_loss, valloss, epoch, i)

  checkpoint_was_already_created = save_iteration or save_epoch
  if not checkpoint_was_already_created:
    valloss = validate(model, criterion, valset, iteration - 1, hparams.batch_size, collate_fn, logger)
    #if rank == 0:
    save_checkpoint(model, optimizer, learning_rate, iteration - 1, save_checkpoint_dir)
    save_checkpoint_score(save_checkpoint_log_dir, iteration - 1, grad_norm, reduced_loss, valloss, epoch, i)

  debug_logger.info('Finished training.')
  duration_s = time.time() - complete_start
  duration_m = duration_s / 60
  debug_logger.info('Duration: {:.2f}min'.format(duration_m))


def continue_train(hparams: str, n_symbols: int, n_speakers: int, logdir: str, trainset: PreparedDataList, valset: PreparedDataList, save_checkpoint_dir: str, save_checkpoint_log_dir: str):
  hp = create_hparams(hparams)

  hp.n_symbols = n_symbols
  hp.n_speakers = n_speakers
  last_checkpoint = get_last_checkpoint(save_checkpoint_dir)
  assert last_checkpoint
  last_checkpoint_path = os.path.join(save_checkpoint_dir, last_checkpoint)

  model, optimizer, learning_rate, iteration = _train("", "", last_checkpoint_path, hp)
  logger = Tacotron2Logger(logdir)
  train_core(hp, logger, trainset, valset, save_checkpoint_dir, save_checkpoint_log_dir, iteration, model, optimizer, learning_rate)


def train(warm_start_model_path: str, weights_path: str, hparams: str, logdir: str, n_symbols: int, n_speakers: int, trainset: PreparedDataList, valset: PreparedDataList, save_checkpoint_dir: str, save_checkpoint_log_dir: str):
  hp = create_hparams(hparams)

  hp.n_symbols = n_symbols
  hp.n_speakers = n_speakers

  model, optimizer, learning_rate, iteration = _train(warm_start_model_path, weights_path, "", hp)
  logger = Tacotron2Logger(logdir)
  train_core(hp, logger, trainset, valset, save_checkpoint_dir, save_checkpoint_log_dir, iteration, model, optimizer, learning_rate)


def _train(warm_start_model_path: str, weights_path: str, checkpoint_path: str, hparams):
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

  torch.backends.cudnn.enabled = hparams.cudnn_enabled
  torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

  torch.manual_seed(hparams.seed)
  torch.cuda.manual_seed(hparams.seed)

  model = load_model(hparams)
  learning_rate = hparams.learning_rate
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=hparams.weight_decay)

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
    if weights_path:
      debug_logger.info("Init weights from '{}'".format(weights_path))
      weights = load_weights(weights_path)
      init_weights(model, weights)

  return model, optimizer, learning_rate, iteration

def load_checkpoint(checkpoint_path, model, optimizer):
  assert os.path.isfile(checkpoint_path)
  debug_logger.info("Loading checkpoint '{}'".format(checkpoint_path))
  checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
  model.load_state_dict(checkpoint_dict['state_dict'])
  optimizer.load_state_dict(checkpoint_dict['optimizer'])
  learning_rate = checkpoint_dict['learning_rate']
  iteration = checkpoint_dict['iteration']
  debug_logger.info("Loaded checkpoint '{}' from iteration {}".format(checkpoint_path, iteration))
  return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, parent_dir):
  filepath = os.path.join(parent_dir, f"{iteration}.pt")
  debug_logger.info("Saving model and optimizer state at iteration {} to {}".format(iteration, filepath))
  torch.save(
    {
      'iteration': iteration,
      'state_dict': model.state_dict(),
      'optimizer': optimizer.state_dict(),
      'learning_rate': learning_rate
    },
    filepath
  )

def save_checkpoint_score(parent_dir, iteration, gradloss, trainloss, valloss, epoch, i):
  filepath = os.path.join(parent_dir, str(iteration))
  loss_avg = (trainloss + valloss) / 2
  name = "{}_epoch-{}_it-{}_grad-{:.6f}_train-{:.6f}_val-{:.6f}_avg-{:.6f}.log".format(filepath, epoch, i, gradloss, trainloss, valloss, loss_avg)
  with open(name, mode='w', encoding='utf-8') as f:
    f.write("Training Grad Norm: {:.6f}\nTraining Loss: {:.6f}\nValidation Loss: {:.6f}".format(gradloss, trainloss, valloss))

def load_weights(weights_path: str):
  assert os.path.isfile(weights_path)
  weights = np.load(weights_path)
  weights = torch.from_numpy(weights)
  return weights

def init_weights(model, weights):
  dummy_dict = model.state_dict()
  update = { 'embedding.weight': weights }
  dummy_dict.update(update)
  model_dict = dummy_dict
  model.load_state_dict(model_dict)
