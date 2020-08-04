# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#    * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#    * Neither the name of the NVIDIA CORPORATION nor the
#    names of its contributors may be used to endorse or promote products
#    derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************
import os
import time

import torch
from src.common.train_log import log, reset_log
from src.common.utils import (args_to_str, get_last_checkpoint,
                              get_total_duration_min_df)
from src.paths import get_checkpoint_dir, get_log_dir
from src.waveglow.data_utils import MelLoader
from src.waveglow.hparams import create_hparams
from src.waveglow.model import WaveGlow, WaveGlowLoss
from src.waveglow.prepare_ds import prepare
from src.waveglow.prepare_ds_io import parse_traindata
from torch.utils.data import DataLoader

# #=====START: ADDED FOR DISTRIBUTED======
# from distributed_waveglow import init_distributed, apply_gradient_allreduce, reduce_tensor
# from torch.utils.data.distributed import DistributedSampler
# #=====END:   ADDED FOR DISTRIBUTED======

def load_model(hparams):
  model = WaveGlow(hparams)
  model = model.cuda()

  return model

def load_checkpoint(checkpoint_path, model, optimizer):
  assert os.path.isfile(checkpoint_path)
  checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
  iteration = checkpoint_dict['iteration']
  learning_rate = checkpoint_dict['learning_rate']
  optimizer_state_dict = checkpoint_dict['optimizer']
  optimizer.load_state_dict(optimizer_state_dict)
  # model_for_loading = checkpoint_dict['model']
  # model.load_state_dict(model_for_loading.state_dict())
  model_state_dict = checkpoint_dict['state_dict']
  model.load_state_dict(model_state_dict)
  print("Loaded checkpoint '{}' (iteration {})" .format(checkpoint_path, iteration))
  return model, optimizer, learning_rate, iteration

def save_checkpoint(model, optimizer, learning_rate, iteration, filepath, hparams):
  print("Saving model and optimizer state at iteration {} to {}".format(iteration, filepath))
  #model_for_saving = WaveGlow(hparams).cuda()
  #model_for_saving.load_state_dict(model.state_dict())

  data = {
    #'model': model_for_saving,
    'state_dict': model.state_dict(),
    'iteration': iteration,
    'optimizer': optimizer.state_dict(),
    'learning_rate': learning_rate
  }

  torch.save(data, filepath)

def train(training_dir_path, hparams, rank, n_gpus, continue_training: bool):
  torch.manual_seed(hparams.seed)
  torch.cuda.manual_seed(hparams.seed)
  # #=====START: ADDED FOR DISTRIBUTED======
  # if n_gpus > 1:
  #   init_distributed(rank, n_gpus, group_name, **dist_config)
  # #=====END:   ADDED FOR DISTRIBUTED======

  criterion = WaveGlowLoss(
    sigma=hparams.sigma
  )

  model = load_model(hparams)

  # #=====START: ADDED FOR DISTRIBUTED======
  # if n_gpus > 1:
  #   model = apply_gradient_allreduce(model)
  # #=====END:   ADDED FOR DISTRIBUTED======

  optimizer = torch.optim.Adam(model.parameters(), lr=hparams.learning_rate)

  # if fp16_run:
  #   from apex import amp
  #   model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

  # Load checkpoint if one exists
  iteration = 0
  if continue_training:
    checkpoint_dir = get_checkpoint_dir(training_dir_path)
    last_checkpoint = get_last_checkpoint(checkpoint_dir)

    if not last_checkpoint:
      raise Exception("No checkpoint was found to continue training!")

    full_checkpoint_path = os.path.join(checkpoint_dir, last_checkpoint)
    log(training_dir_path, "Loading checkpoint '{}'".format(full_checkpoint_path))
    model, optimizer, learning_rate, iteration = load_checkpoint(full_checkpoint_path, model, optimizer)
    log(training_dir_path, "Loaded checkpoint '{}' from iteration {}" .format(full_checkpoint_path, iteration))
    iteration += 1  # next iteration is iteration + 1
  
  #filelist_dir_path = get_filelist_dir(training_dir_path)
  # trainset_path = os.path.join(filelist_dir_path, filelist_training_file_name)
  # train_dur = get_total_duration_min_df(trainset_path, duration_column=duration_col)
  # print("Duration trainset {:.2f}min / {:.2f}h".format(train_dur, train_dur / 60))
  trainset = MelLoader(prepare_ds_data=parse_traindata(training_dir_path), hparams=hparams)
  train_sampler = None
  # # =====START: ADDED FOR DISTRIBUTED======
  # train_sampler = DistributedSampler(trainset) if n_gpus > 1 else None
  # # =====END:   ADDED FOR DISTRIBUTED======
  train_loader = DataLoader(
    trainset,
    num_workers=1,
    shuffle=False,
    sampler=train_sampler,
    batch_size=hparams.batch_size,
    pin_memory=False,
    drop_last=True
  )

  # Get shared output_directory ready
  # if rank == 0:
  #   if not os.path.isdir(output_directory):
  #     os.makedirs(output_directory)
  #     os.chmod(output_directory, 0o775)
  #   print("output directory", output_directory)

  if hparams.with_tensorboard and rank == 0:
    from tensorboardX import SummaryWriter
    logger = SummaryWriter(get_log_dir(training_dir_path))

  model.train()
  epoch_offset = max(0, int(iteration / len(train_loader)))
  # ================ MAIN TRAINING LOOP! ===================
  for epoch in range(epoch_offset, hparams.epochs):
    print("Epoch: {}".format(epoch))
    for i, batch in enumerate(train_loader):
      model.zero_grad()

      mel, audio = batch
      mel = torch.autograd.Variable(mel.cuda())
      audio = torch.autograd.Variable(audio.cuda())
      outputs = model((mel, audio))

      loss = criterion(outputs)
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

      print("{}:\t{:.9f}".format(iteration, reduced_loss))
      if hparams.with_tensorboard and rank == 0:
        logger.add_scalar('training_loss', reduced_loss, i + len(train_loader) * epoch)

      if (iteration % hparams.iters_per_checkpoint == 0):
        if rank == 0:
          checkpoint_dir = get_checkpoint_dir(training_dir_path)
          checkpoint_path = os.path.join(checkpoint_dir,  str(iteration))
          save_checkpoint(model, optimizer, hparams.learning_rate, iteration, checkpoint_path, hparams)

      iteration += 1

def start_train(training_dir_path: str, hparams, continue_training: bool):
  start = time.time()

  log(training_dir_path, 'Final parsed hparams:')
  log(training_dir_path, '\n'.join(str(hparams.values()).split(',')))

  torch.backends.cudnn.enabled = True
  torch.backends.cudnn.benchmark = False
  # torch.backends.cudnn.enabled = hparams.cudnn_enabled
  # torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

  log(training_dir_path, "Epochs: {}".format(hparams.epochs))
  log(training_dir_path, "Batchsize: {}".format(hparams.batch_size))
  log(training_dir_path, "FP16 Run: {}".format(hparams.fp16_run))
  #log(training_dir_path, "Dynamic Loss Scaling: {}".format(hparams.dynamic_loss_scaling))
  log(training_dir_path, "Distributed Run: {}".format(False))
  log(training_dir_path, "cuDNN Enabled: {}".format(torch.backends.cudnn.enabled))
  log(training_dir_path, "cuDNN Benchmark: {}".format(torch.backends.cudnn.benchmark))

  rank = 0 # 'rank of current gpu'
  n_gpus = torch.cuda.device_count() # 'number of gpus'
  if n_gpus > 1:
    raise Exception("More than one GPU is currently not supported.")
  #group_name = "group_name" # 'Distributed group name'
  # if n_gpus > 1:
  #   if args.group_name == '':
  #     print("WARNING: Multiple GPUs detected but no distributed group set")
  #     print("Only running 1 GPU.  Use distributed.py for multiple GPUs")
  #     n_gpus = 1
  # if n_gpus == 1 and args.rank != 0:
  #   raise Exception("Doing single GPU training on rank > 0")

  train(training_dir_path=training_dir_path, n_gpus=n_gpus, rank=rank, hparams=hparams, continue_training=continue_training)

  log(training_dir_path, 'Finished training.')
  duration_s = time.time() - start
  duration_m = duration_s / 60
  log(training_dir_path, 'Duration: {:.2f}min'.format(duration_m))

def init_train_parser(parser):
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--training_dir', type=str, required=True)
  parser.add_argument('--wav_ds_name', type=str, required=True)
  parser.add_argument('--test_size', type=float, default=0.001)
  parser.add_argument('--validation_size', type=float, default=0.01)
  parser.add_argument('--hparams', type=str)
  parser.add_argument('--continue_training', action='store_true')
  return __main

def __main(base_dir, training_dir, continue_training, wav_ds_name, test_size, validation_size, hparams):
  hparams = create_hparams(hparams)
  training_dir_path = os.path.join(base_dir, training_dir)

  if not continue_training:
    reset_log(training_dir_path)

    prepare(
      base_dir=base_dir,
      training_dir_path=training_dir_path,
      wav_ds_name=wav_ds_name,
      test_size=test_size,
      validation_size=validation_size,
      seed=hparams.seed,
    )
    
  start_train(
    training_dir_path=training_dir_path,
    hparams=hparams,
    continue_training=continue_training
  )

if __name__ == "__main__":
  __main(
    base_dir = '/datasets/models/taco2pt_v2',
    training_dir = 'wg_debug',
    wav_ds_name = 'ljs_22050kHz',
    test_size = 0.01,
    validation_size = 0.5,
    hparams = 'batch_size=4,iters_per_checkpoint=5,fp16_run=False,with_tensorboard=True',
    continue_training = False,
  )
