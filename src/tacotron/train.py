import argparse
import json
import math
import os
import time
from shutil import copyfile

import numpy as np
from numpy import finfo

import torch
#from distributed_tacotron import apply_gradient_allreduce
import torch.distributed as dist
from src.common.train_log import log, reset_log
from src.common.utils import (args_to_str, duration_col, get_last_checkpoint,
                              get_total_duration_min_df, parse_ds_speakers,
                              parse_json)
from src.paths import (ds_preprocessed_file_name, ds_preprocessed_symbols_name,
                       filelist_file_name, filelist_speakers_name,
                       filelist_symbols_file_name, filelist_training_file_name,
                       filelist_validation_file_name,
                       filelist_weights_file_name, get_checkpoint_dir,
                       get_ds_dir, get_filelist_dir, get_log_dir,
                       get_symbols_path, inference_config_file,
                       log_inference_config, log_input_file, log_map_file,
                       log_train_config, log_train_map, train_config_file,
                       train_map_file)
from src.tacotron.data_utils import SymbolsMelCollate, SymbolsMelLoader
from src.tacotron.hparams import create_hparams
from src.tacotron.logger import Tacotron2Logger
from src.tacotron.loss_function import Tacotron2Loss
from src.tacotron.model import Tacotron2
from src.tacotron.plot_embeddings import analyse
from src.tacotron.prepare_ds import prepare
from src.tacotron.prepare_ds_ms import prepare as prepare_ms
from src.tacotron.txt_pre import process_input_text
from src.text.symbol_converter import load_from_file
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from src.tacotron.prepare_ds_ms_io import parse_traindata, parse_validationset

def reduce_tensor(tensor, n_gpus):
  rt = tensor.clone()
  dist.all_reduce(rt, op=dist.reduce_op.SUM)
  rt /= n_gpus
  return rt


def init_distributed(hparams, n_gpus, rank, group_name, training_dir_path):
  assert torch.cuda.is_available(), "Distributed mode requires CUDA."
  log(training_dir_path, "Initializing Distributed")

  # Set cuda device so everything is done on the right GPU.
  torch.cuda.set_device(rank % torch.cuda.device_count())

  # Initialize distributed communication
  dist.init_process_group(
    backend=hparams.dist_backend, init_method=hparams.dist_url,
    world_size=n_gpus, rank=rank, group_name=group_name)

  log(training_dir_path, "Done initializing distributed")


def prepare_dataloaders(hparams, training_dir_path):
  # Get data, data loaders and collate function ready
  # trainset_path = os.path.join(filelist_dir_path, filelist_training_file_name)
  # train_dur = get_total_duration_min_df(trainset_path)
  # print("Duration trainset {:.2f}min / {:.2f}h".format(train_dur, train_dur / 60))
  
  # valset_path = os.path.join(filelist_dir_path, filelist_validation_file_name)
  # val_dur = get_total_duration_min_df(valset_path)
  # print("Duration valset {:.2f}min / {:.2f}h".format(val_dur, val_dur / 60))
  # valset = SymbolsMelLoader(valset_path, hparams)

  traindata = parse_traindata(training_dir_path)
  trainset = SymbolsMelLoader(traindata, hparams)

  valdata = parse_validationset(training_dir_path)
  valset = SymbolsMelLoader(valdata, hparams)

  collate_fn = SymbolsMelCollate(hparams.n_frames_per_step)

  if hparams.distributed_run:
    train_sampler = DistributedSampler(trainset)
    shuffle = False
  else:
    train_sampler = None
    shuffle = True

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
  # if hparams.fp16_run:
  #   model.decoder.attention_layer.score_mask_value = finfo('float16').min

  # if hparams.distributed_run:
  #   model = apply_gradient_allreduce(model)

  return model

def warm_start_model(checkpoint_path, model, ignore_layers, training_dir_path):
  assert os.path.isfile(checkpoint_path)
  log(training_dir_path, "Warm starting model from checkpoint '{}'".format(checkpoint_path))
  checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
  model_dict = checkpoint_dict['state_dict']
  if len(ignore_layers) > 0:
    model_dict = {k: v for k, v in model_dict.items() if k not in ignore_layers}
    dummy_dict = model.state_dict()
    dummy_dict.update(model_dict)
    model_dict = dummy_dict
  model.load_state_dict(model_dict)

def init_weights(weights_path, model, training_dir_path):
  assert os.path.isfile(weights_path)
  log(training_dir_path, "Init weights from '{}'".format(weights_path))
  weights = np.load(weights_path)
  weights = torch.from_numpy(weights)
  dummy_dict = model.state_dict()
  update = { 'embedding.weight': weights }
  dummy_dict.update(update)
  model_dict = dummy_dict
  model.load_state_dict(model_dict)

def load_checkpoint(checkpoint_path, model, optimizer, training_dir_path):
  #weights_path = os.path.join(speaker_dir, weights_name)
  #assert os.path.isfile(weights_path)
  assert os.path.isfile(checkpoint_path)
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
  return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, filepath, training_dir_path):
  log(training_dir_path, "Saving model and optimizer state at iteration {} to {}".format(iteration, filepath))
  torch.save(
    {
      'iteration': iteration,
      'state_dict': model.state_dict(),
      'optimizer': optimizer.state_dict(),
      'learning_rate': learning_rate
    },
    filepath
  )

def save_checkpoint_score(checkpoint_path, gradloss, trainloss, valloss, epoch, i):
  loss_avg = (trainloss + valloss) / 2
  name = "{}_epoch-{}_it-{}_grad-{:.6f}_train-{:.6f}_val-{:.6f}_avg-{:.6f}.log".format(checkpoint_path, epoch, i, gradloss, trainloss, valloss, loss_avg)
  with open(name, mode='w', encoding='utf-8') as f:
    f.write("Training Grad Norm: {:.6f}\nTraining Loss: {:.6f}\nValidation Loss: {:.6f}".format(gradloss, trainloss, valloss))

def validate_core(model, criterion, valset, batch_size, n_gpus,
       collate_fn, distributed_run):
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
  return val_loss, model, y, y_pred

def validate(model, criterion, valset, iteration, batch_size, n_gpus,
       collate_fn, logger, distributed_run, rank, training_dir_path):
  val_loss, model, y, y_pred = validate_core(model, criterion, valset, batch_size, n_gpus, collate_fn, distributed_run)

  if rank == 0:
    log(training_dir_path, "Validation loss {}: {:9f}  ".format(iteration, val_loss))
    logger.log_validation(val_loss, model, y, y_pred, iteration)
  
  return val_loss

def train(pretrained_path, use_weights: bool, warm_start, n_gpus,
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
    init_distributed(hparams, n_gpus, rank, group_name, training_dir_path)

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
  train_loader, valset, collate_fn = prepare_dataloaders(hparams, training_dir_path)

  if not len(train_loader):
    print("Not enough trainingdata.")
    return False

  # Load checkpoint if one exists
  iteration = 0
  epoch_offset = 0

  if continue_training:
    checkpoint_dir = get_checkpoint_dir(training_dir_path)
    last_checkpoint = get_last_checkpoint(checkpoint_dir)

    if not last_checkpoint:
      raise Exception("No checkpoint was found to continue training!")

    full_checkpoint_path = os.path.join(get_checkpoint_dir(training_dir_path), last_checkpoint)

    log(training_dir_path, "Loading checkpoint '{}'".format(full_checkpoint_path))
    
    model, optimizer, _learning_rate, iteration = load_checkpoint(full_checkpoint_path, model, optimizer, training_dir_path)

    log(training_dir_path, "Loaded checkpoint '{}' from iteration {}" .format(full_checkpoint_path, iteration))

    if hparams.use_saved_learning_rate:
      learning_rate = _learning_rate
    iteration += 1  # next iteration is iteration + 1
    epoch_offset = max(0, int(iteration / len(train_loader)))
  else:
    if warm_start:
      assert pretrained_path
      # raise Exception("Warm start was not possible because the path to the model was not valid.")
      warm_start_model(pretrained_path, model, hparams.ignore_layers, training_dir_path)
    
    if use_weights:
      weight_file = os.path.join(filelist_dir_path, filelist_weights_file_name)
      init_weights(weight_file, model, training_dir_path)

  log(training_dir_path, "Modelweights:")
  log(training_dir_path, str(model.state_dict()['embedding.weight']))
  log(training_dir_path, "Iterations per epoch: {}".format(len(train_loader)))

  model.train()
  is_overflow = False
  total_its = hparams.epochs * len(train_loader)
  # ================ MAIN TRAINING LOOP! ===================
  for epoch in range(epoch_offset, hparams.epochs):
    log(training_dir_path, "Epoch: {}".format(epoch))
    
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

      #if not is_overflow and rank == 0:
      duration = time.perf_counter() - start
      log(training_dir_path, "Epoch: {}/{} | Iteration: {}/{} | Total iteration: {}/{} | Train loss: {:.6f} | Grad Norm: {:.6f} | Duration: {:.2f}s/it".format(
        str(epoch).zfill(len(str(hparams.epochs))),
        hparams.epochs,
        str(i).zfill(len(str(len(train_loader) - 1))),
        len(train_loader) - 1,
        str(iteration).zfill(len(str(total_its))),
        total_its,
        reduced_loss,
        grad_norm,
        duration
      ))
      logger.log_training(reduced_loss, grad_norm, learning_rate, duration, iteration)

      #if not is_overflow and (iteration % hparams.iters_per_checkpoint == 0):
      save_iteration = (hparams.iters_per_checkpoint > 0 and (iteration % hparams.iters_per_checkpoint == 0)) or iteration == 0
      if save_iteration:
        valloss = validate(model, criterion, valset, iteration, hparams.batch_size, n_gpus, collate_fn, logger, hparams.distributed_run, rank, training_dir_path)
        #if rank == 0:
        checkpoint_path = os.path.join(output_directory, str(iteration))
        save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path, training_dir_path)
        save_checkpoint_score(checkpoint_path, grad_norm, reduced_loss, valloss, epoch, i)
      iteration += 1

    checkpoint_was_already_created = save_iteration
    save_epoch = not checkpoint_was_already_created and hparams.epochs_per_checkpoint > 0 and (epoch % hparams.epochs_per_checkpoint == 0)
    if save_epoch:
      valloss = validate(model, criterion, valset, iteration - 1, hparams.batch_size, n_gpus, collate_fn, logger, hparams.distributed_run, rank, training_dir_path)
      #if rank == 0:
      checkpoint_path = os.path.join(output_directory, str(iteration - 1))
      save_checkpoint(model, optimizer, learning_rate, iteration - 1, checkpoint_path, training_dir_path)
      save_checkpoint_score(checkpoint_path, grad_norm, reduced_loss, valloss, epoch, i)

  checkpoint_was_already_created = save_iteration or save_epoch
  if not checkpoint_was_already_created:
    valloss = validate(model, criterion, valset, iteration - 1, hparams.batch_size, n_gpus, collate_fn, logger, hparams.distributed_run, rank, training_dir_path)
    #if rank == 0:
    checkpoint_path = os.path.join(output_directory,  str(iteration - 1))
    save_checkpoint(model, optimizer, learning_rate, iteration - 1, checkpoint_path, training_dir_path)
    save_checkpoint_score(checkpoint_path, grad_norm, reduced_loss, valloss, epoch, i)
  return True

def start_train(training_dir_path: str, hparams, use_weights: str, pretrained_path: str, warm_start: bool, continue_training: bool):
  start = time.time()
  conv = load_from_file(get_symbols_path(training_dir_path))
  hparams.n_symbols = conv.get_symbol_ids_count()

  speakers_file = os.path.join(get_filelist_dir(training_dir_path), filelist_speakers_name)
  all_speakers = parse_json(speakers_file)
  hparams.n_speakers = len(all_speakers)
  
  log(training_dir_path, 'Final parsed hparams:')
  x = '\n'.join(str(hparams.values()).split(','))
  log(training_dir_path, x)

  torch.backends.cudnn.enabled = hparams.cudnn_enabled
  torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

  log(training_dir_path, "Epochs: {}".format(hparams.epochs))
  log(training_dir_path, "Batchsize: {}".format(hparams.batch_size))
  log(training_dir_path, "FP16 Run: {}".format(hparams.fp16_run))
  log(training_dir_path, "Dynamic Loss Scaling: {}".format(hparams.dynamic_loss_scaling))
  log(training_dir_path, "Distributed Run: {}".format(hparams.distributed_run))
  log(training_dir_path, "cuDNN Enabled: {}".format(hparams.cudnn_enabled))
  log(training_dir_path, "cuDNN Benchmark: {}".format(hparams.cudnn_benchmark))

  rank = 0 # 'rank of current gpu'
  n_gpus = 1 # 'number of gpus'
  group_name = "group_name" # 'Distributed group name'

  successfull = train(pretrained_path, use_weights, warm_start, n_gpus, rank, group_name, hparams, continue_training, training_dir_path)
  if successfull:
    log(training_dir_path, 'Finished training.')
    duration_s = time.time() - start
    duration_m = duration_s / 60
    log(training_dir_path, 'Duration: {:.2f}min'.format(duration_m))
# #   #hparams.batch_size=22 only when on all speakers simultanously thchs

from argparse import ArgumentParser
def init_train_parser(parser: ArgumentParser):
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--training_dir', type=str, required=True)
  parser.add_argument('--continue_training', action='store_true')
  parser.add_argument('--warm_start', action='store_true')
  parser.add_argument('--warm_start_model', type=str)
  parser.add_argument('--speakers', type=str, help="ds_name,speaker_id;... or ds_name,all;...", required=True)
  parser.add_argument('--test_size', type=float, default=0.001)
  parser.add_argument('--validation_size', type=float, default=0.1)
  parser.add_argument('--hparams', type=str)
  parser.add_argument('--weight_map_mode', type=str, choices=['same_symbols_only', 'use_map'])
  parser.add_argument('--weight_map', type=str)
  parser.add_argument('--weight_map_model', type=str)
  parser.add_argument('--weight_map_model_symbols', type=str)
  return __main

def __main(base_dir, training_dir, continue_training, warm_start, warm_start_model, speakers, test_size, validation_size, hparams, weight_map_model, weight_map_model_symbols, weight_map_mode, weight_map):
  hparams = create_hparams(hparams)
  training_dir_path = os.path.join(base_dir, training_dir)

  if not continue_training:
    use_map = weight_map_mode == 'use_map'
    map_path = os.path.join(training_dir_path, train_map_file)
    if use_map:
      log_train_map(training_dir_path, weight_map)
    elif os.path.exists(map_path):
      os.remove(map_path)

    reset_log(training_dir_path)
    prepare_ms(base_dir, training_dir_path, speakers=speakers, pretrained_model=weight_map_model, weight_map_mode=weight_map_mode, hparams=hparams, pretrained_model_symbols=weight_map_model_symbols, test_size=test_size, val_size=validation_size, seed=hparams.seed)
    
  weights_path = os.path.join(get_filelist_dir(training_dir_path), filelist_weights_file_name)
  use_weights_map = os.path.exists(weights_path)
  successfull_training = start_train(training_dir_path, hparams=hparams, use_weights=use_weights_map, pretrained_path=warm_start_model, warm_start=warm_start, continue_training=continue_training)
  if successfull_training:
    analyse(training_dir_path)


if __name__ == "__main__":
  __main(
    base_dir = '/datasets/models/taco2pt_v2',
    training_dir = 'debug',
    continue_training = False,
    warm_start = True,
    warm_start_model = "/datasets/models/pretrained/ljs_ipa_scratch_80000",
    speakers = 'thchs_mel_v1,D31',
    test_size = 0.1,
    validation_size = 0.05,
    hparams = 'batch_size=17,iters_per_checkpoint=0,epochs_per_checkpoint=1,ignore_layers=[embedding.weight, speakers_embedding.weight]',
    weight_map_model = None,
    weight_map_model_symbols = None,
    weight_map_mode = None,
    weight_map = None,
    #weight_map_mode = 'same_symbols_only',
    #weight_map_mode = 'use_map',
    #map = "maps/weights/chn_en_v1.json",
    #pretrained_model = "/datasets/models/pretrained/ljs_ipa_scratch_80000",
    #pretrained_model_symbols = "/datasets/models/pretrained/ljs_ipa_scratch.json",
    #continue_training = True,
  )
