import os
import time
import argparse
import math
from numpy import finfo
import numpy as np

import torch
from distributed import apply_gradient_allreduce
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from model import Tacotron2
from data_utils import SymbolsMelLoader, SymbolsMelCollate
from loss_function import Tacotron2Loss
from logger import Tacotron2Logger
from hparams import create_hparams

from text.symbol_converter import load_from_file
from paths import filelist_training_file_name, filelist_validation_file_name

def reduce_tensor(tensor, n_gpus):
  rt = tensor.clone()
  dist.all_reduce(rt, op=dist.reduce_op.SUM)
  rt /= n_gpus
  return rt


def init_distributed(hparams, n_gpus, rank, group_name):
  assert torch.cuda.is_available(), "Distributed mode requires CUDA."
  print("Initializing Distributed")

  # Set cuda device so everything is done on the right GPU.
  torch.cuda.set_device(rank % torch.cuda.device_count())

  # Initialize distributed communication
  dist.init_process_group(
    backend=hparams.dist_backend, init_method=hparams.dist_url,
    world_size=n_gpus, rank=rank, group_name=group_name)

  print("Done initializing distributed")


def prepare_dataloaders(hparams, filelist_dir_path):
  # Get data, data loaders and collate function ready
  trainset = SymbolsMelLoader(os.path.join(filelist_dir_path, filelist_training_file_name), hparams)
  valset = SymbolsMelLoader(os.path.join(filelist_dir_path, filelist_validation_file_name), hparams)
  collate_fn = SymbolsMelCollate(hparams.n_frames_per_step)

  if hparams.distributed_run:
    train_sampler = DistributedSampler(trainset)
    shuffle = False
  else:
    train_sampler = None
    shuffle = True

  train_loader = DataLoader(trainset, num_workers=1, shuffle=shuffle,
                sampler=train_sampler,
                batch_size=hparams.batch_size, pin_memory=False,
                drop_last=True, collate_fn=collate_fn)
  return train_loader, valset, collate_fn


def prepare_directories_and_logger(output_directory, log_directory, rank):
  if rank == 0:
    if not os.path.isdir(output_directory):
      os.makedirs(output_directory)
      os.chmod(output_directory, 0o775)
    logger = Tacotron2Logger(os.path.join(output_directory, log_directory))
  else:
    logger = None
  return logger


def load_model(hparams):
  model = Tacotron2(hparams).cuda()
  if hparams.fp16_run:
    model.decoder.attention_layer.score_mask_value = finfo('float16').min

  if hparams.distributed_run:
    model = apply_gradient_allreduce(model)

  return model


def warm_start_model(checkpoint_path, model, ignore_layers):
  assert os.path.isfile(checkpoint_path)
  print("Warm starting model from checkpoint '{}'".format(checkpoint_path))
  checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
  model_dict = checkpoint_dict['state_dict']
  if len(ignore_layers) > 0:
    model_dict = {k: v for k, v in model_dict.items()
            if k not in ignore_layers}
    dummy_dict = model.state_dict()
    dummy_dict.update(model_dict)
    model_dict = dummy_dict
  model.load_state_dict(model_dict)
  return model

def init_weights(weights_path, model):
  assert os.path.isfile(weights_path)
  print("Init weights from '{}'".format(weights_path))
  weights = np.load(weights_path)
  weights = torch.from_numpy(weights)
  dummy_dict = model.state_dict()
  update = { 'embedding.weight': weights }
  dummy_dict.update(update)
  model_dict = dummy_dict
  model.load_state_dict(model_dict)
  return model


def load_checkpoint(checkpoint_path, model, optimizer):
  #weights_path = os.path.join(speaker_dir, weights_name)
  #assert os.path.isfile(weights_path)
  assert os.path.isfile(checkpoint_path)
  print("Loading checkpoint '{}'".format(checkpoint_path))
  checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
  ### Didn't worked out bc the optimizer has old weight size
  # if overwrite_weights:
  #   weights = np.load(weights_path)
  #   weights = torch.from_numpy(weights)
  #   dummy_dict = model.state_dict()
  #   update = { 
  #       'embedding.weight': weights 
  #   }
  #   checkpoint_dict.update({'iteration':0})
  #   y_ref = weights[0]
  #   x = checkpoint_dict['state_dict']['embedding.weight'][0]
  #   checkpoint_dict['state_dict'].update(update)
  #   y = checkpoint_dict['state_dict']['embedding.weight'][0]
  #   #checkpoint_dict['state_dict']['embedding.weights'] = weights
  model.load_state_dict(checkpoint_dict['state_dict'])
  optimizer.load_state_dict(checkpoint_dict['optimizer'])
  learning_rate = checkpoint_dict['learning_rate']
  iteration = checkpoint_dict['iteration']
  print("Loaded checkpoint '{}' from iteration {}" .format(
    checkpoint_path, iteration))
  return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
  print("Saving model and optimizer state at iteration {} to {}".format(
    iteration, filepath))
  torch.save({'iteration': iteration,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'learning_rate': learning_rate}, filepath)


def validate(model, criterion, valset, iteration, batch_size, n_gpus,
       collate_fn, logger, distributed_run, rank):
  """Handles all the validation scoring and printing"""
  model.eval()
  with torch.no_grad():
    val_sampler = DistributedSampler(valset) if distributed_run else None
    val_loader = DataLoader(valset, sampler=val_sampler, num_workers=1,
                shuffle=False, batch_size=batch_size,
                pin_memory=False, collate_fn=collate_fn)

    val_loss = 0.0
    for i, batch in enumerate(val_loader):
      x, y = model.parse_batch(batch)
      y_pred = model(x)
      loss = criterion(y_pred, y)
      if distributed_run:
        reduced_val_loss = reduce_tensor(loss.data, n_gpus).item()
      else:
        reduced_val_loss = loss.item()
      val_loss += reduced_val_loss
    val_loss = val_loss / (i + 1)

  model.train()
  if rank == 0:
    print("Validation loss {}: {:9f}  ".format(iteration, val_loss))
    logger.log_validation(val_loss, model, y, y_pred, iteration)

from paths import get_symbols_path, get_filelist_dir, get_checkpoint_dir, get_log_dir, filelist_weights_file_name

def train(pretrained_path, use_pretrained_weights: bool, warm_start, n_gpus,
      rank, group_name, hparams, continue_training, training_dir_path):
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
  if hparams.distributed_run:
    init_distributed(hparams, n_gpus, rank, group_name)

  torch.manual_seed(hparams.seed)
  torch.cuda.manual_seed(hparams.seed)

  model = load_model(hparams)
  learning_rate = hparams.learning_rate
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=hparams.weight_decay)

  # if hparams.fp16_run:
  #   from apex import amp
  #   model, optimizer = amp.initialize(model, optimizer, opt_level='O2')

  if hparams.distributed_run:
    model = apply_gradient_allreduce(model)

  criterion = Tacotron2Loss()

  output_directory = get_checkpoint_dir(training_dir_path)
  log_directory = get_log_dir(training_dir_path)
  filelist_dir_path = get_filelist_dir(training_dir_path)

  logger = prepare_directories_and_logger(output_directory, log_directory, rank)
  train_loader, valset, collate_fn = prepare_dataloaders(hparams, filelist_dir_path)

  # Load checkpoint if one exists
  iteration = 0
  epoch_offset = 0
  checkpoint_loaded = False
  if continue_training:
    full_checkpoint_path = ''

    # TODO try get last iteration

    if full_checkpoint_path != '':
      model, optimizer, _learning_rate, iteration = load_checkpoint(full_checkpoint_path, model, optimizer)
      if hparams.use_saved_learning_rate:
        learning_rate = _learning_rate
      iteration += 1  # next iteration is iteration + 1
      epoch_offset = max(0, int(iteration / len(train_loader)))
      checkpoint_loaded = True
  
  if not checkpoint_loaded and pretrained_path != None and warm_start:
    model = warm_start_model(pretrained_path, model, hparams.ignore_layers)
  
  # if checkpoint_path is not None:
  #   full_checkpoint_path = os.path.join(base_dir, checkpoint_path)
  #   if warm_start:
  #     model = warm_start_model(pretrained_path, model, hparams.ignore_layers)
  #     if use_pretrained_weights:
  #       init_weights(speaker_dir, model)
  #   else:
  #     model, optimizer, _learning_rate, iteration = load_checkpoint(full_checkpoint_path, model, optimizer, speaker_dir)
  #     if hparams.use_saved_learning_rate:
  #       learning_rate = _learning_rate
  #     iteration += 1  # next iteration is iteration + 1
  #     epoch_offset = max(0, int(iteration / len(train_loader)))
  # else:
  
  if not checkpoint_loaded and use_pretrained_weights:
    weight_file = os.path.join(filelist_dir_path, filelist_weights_file_name)
    init_weights(weight_file, model)

  model.train()
  is_overflow = False
  # ================ MAIN TRAINNIG LOOP! ===================
  for epoch in range(epoch_offset, hparams.epochs):
    print("Epoch: {}".format(epoch))
    for i, batch in enumerate(train_loader):
      start = time.perf_counter()
      for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

      model.zero_grad()
      x, y = model.parse_batch(batch)
      y_pred = model(x)

      loss = criterion(y_pred, y)
      if hparams.distributed_run:
        reduced_loss = reduce_tensor(loss.data, n_gpus).item()
      else:
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

      if not is_overflow and rank == 0:
        duration = time.perf_counter() - start
        print("Train loss {} {:.6f} Grad Norm {:.6f} {:.2f}s/it".format(iteration, reduced_loss, grad_norm, duration))
        logger.log_training(reduced_loss, grad_norm, learning_rate, duration, iteration)

      if not is_overflow and (iteration % hparams.iters_per_checkpoint == 0):
        validate(model, criterion, valset, iteration, hparams.batch_size, n_gpus, collate_fn, logger, hparams.distributed_run, rank)
        if rank == 0:
          checkpoint_path = os.path.join(output_directory, "checkpoint_{}".format(iteration))
          save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path)

      iteration += 1

  checkpoint_path = os.path.join(output_directory, "checkpoint_{}".format(iteration - 1))
  save_checkpoint(model, optimizer, learning_rate, iteration - 1, checkpoint_path)

def start_train(base_dir: str, training_dir_path: str, config: dict):
  start = time.time()
  conv = load_from_file(get_symbols_path(training_dir_path))

  hparams = create_hparams(config["hparams"])
  hparams.n_symbols = conv.get_symbol_ids_count()

  torch.backends.cudnn.enabled = hparams.cudnn_enabled
  torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

  print("Epochs:", hparams.epochs)
  print("Batchsize:", hparams.batch_size)
  print("FP16 Run:", hparams.fp16_run)
  print("Dynamic Loss Scaling:", hparams.dynamic_loss_scaling)
  print("Distributed Run:", hparams.distributed_run)
  print("cuDNN Enabled:", hparams.cudnn_enabled)
  print("cuDNN Benchmark:", hparams.cudnn_benchmark)

  rank = 0 # 'rank of current gpu'
  n_gpus = 1 # 'number of gpus'
  group_name = "group_name" # 'Distributed group name'

  train(config["pretrained_path"], config["use_pretrained_weights"], config["warm_start"], n_gpus, rank, group_name, hparams, config["continue_training"], training_dir_path)

  print('Finished training.')
  duration_s = time.time() - start
  duration_m = duration_s / 60
  print('Duration: {:.2f}min'.format(duration_m))


# #args.checkpoint_path = os.path.join(args.base_dir, savecheckpoints_dir, 'checkpoint_49000')
# #args.checkpoint_path = '/datasets/models/pretrained/tacotron2_statedict.pt'
# #args.warm_start = 'false'
# #args.warm_start = 'true'

# debug = str.lower(args.debug) == 'true'

# if debug:
#   args.base_dir = '/datasets/models/taco2pt_ms'
#   weights_path = os.path.join(args.base_dir, weights_name)
#   hparams.sampling_rate = 16000
#   hparams.batch_size = 35
#   hparams.iters_per_checkpoint = 50
#   hparams.epochs = 250 # 250
#   args.checkpoint_path = os.path.join(args.base_dir, savecheckpoints_dir, 'ljs_1_ipa_49000')
#   if False:
#     args.warm_start = 'false'
#     args.use_pretrained_weights = 'true'
#   else:
#     args.warm_start = 'true'
#     args.use_pretrained_weights = 'false'
#     args.checkpoint_path = os.path.join(args.base_dir, savecheckpoints_dir, 'ljs_1_ipa_49000')

# use_pretrained_weights = str.lower(args.use_pretrained_weights) == 'true'


# #hparams.iters_per_checkpoint = 500
# # hparams.epochs = 500

# # # TODO: as param
# # if args.ds_name == "thchs":
# #   # THCHS-30 has 16000
# #   hparams.sampling_rate = 16000
# #   #hparams.batch_size=22 only when on all speakers simultanously
# #   hparams.batch_size=35
# # elif args.ds_name == 'ljs':
# #   hparams.sampling_rate = 22050
# #   hparams.batch_size=26
# # else: 
# #   raise Exception()
