import logging
import os
import random
import time

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.core.common import (TacotronSTFT, get_pytorch_filename,
                             get_last_checkpoint, get_wav_tensor_segment,
                             wav_to_float32_tensor)
from src.core.pre import PreparedData, PreparedDataList
from src.core.waveglow.hparams import create_hparams
from src.core.waveglow.logger import WaveglowLogger
from src.core.waveglow.model import WaveGlow, WaveGlowLoss
import torch

def get_logger():
  return logging.getLogger("wg-train")

debug_logger = get_logger()

class MelLoader(Dataset):
  """
  This is the main class that calculates the spectrogram and returns the
  spectrogram, audio pair.
  """
  def __init__(self, prepare_ds_data: PreparedDataList, hparams):
    self.taco_stft = TacotronSTFT.fromhparams(hparams)
    self.segment_length: int = hparams.segment_length
    self.sampling_rate: int = hparams.sampling_rate
    self.cache_wavs: bool = hparams.cache_wavs

    data = prepare_ds_data
    random.seed(hparams.seed)
    random.shuffle(data)

    wav_paths = {}
    values: PreparedData
    for i, values in enumerate(data):
      wav_paths[i] = values.wav_path
    self.wav_paths = wav_paths

    if hparams.cache_wavs:
      debug_logger.info("Loading wavs into memory...")
      cache = {}
      for i, wav_path in tqdm(wav_paths.items()):
        cache[i] = self._load_wav_tensor(wav_path)
      debug_logger.info("Done")
      self.cache = cache

  def _load_wav_tensor(self, wav_path: str):
    wav_tensor, sr = wav_to_float32_tensor(wav_path)
    if sr != self.sampling_rate:
      raise ValueError("{} {} SR doesn't match target {} SR".format(wav_path, sr, self.sampling_rate))
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

def load_model(hparams):
  model = WaveGlow(hparams)
  model = model.cuda()

  return model

def prepare_dataloaders(hparams, trainset: PreparedDataList, valset: PreparedDataList):
  debug_logger.info(f"Duration trainset {trainset.get_total_duration_s() / 60:.2f}min / {trainset.get_total_duration_s() / 60 / 60:.2f}h")
  debug_logger.info(f"Duration valset {valset.get_total_duration_s() / 60:.2f}min / {valset.get_total_duration_s() / 60 / 60:.2f}h")

  trn = MelLoader(trainset, hparams)

  train_sampler = None
  shuffle = False # maybe set to true bc taco is also true

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

  val = MelLoader(valset, hparams)
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


def validate_core(model, criterion, val_loader: DataLoader):
  """Handles all the validation scoring and printing"""
  model.eval()
  with torch.no_grad():

    # if distributed_run:
    #   val_sampler = DistributedSampler(valset)

    val_loss = 0.0
    debug_logger.info("Validating...")
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

def validate(model, criterion, val_loader, iteration, logger: WaveglowLogger):
  val_loss, model, y_pred = validate_core(model, criterion, val_loader)
  debug_logger.info("Validation loss {}: {:9f}".format(iteration, val_loss))
  logger.log_validation(val_loss, model, y_pred, iteration)

  return val_loss


def train_core(hparams, logdir: str, trainset: PreparedDataList, valset: PreparedDataList, save_checkpoint_dir: str, iteration: int, model, optimizer, learning_rate):
  complete_start = time.time()
  logger = WaveglowLogger(logdir)

  debug_logger.info('Final parsed hparams:')
  debug_logger.info('\n'.join(str(hparams.values()).split(',')))

  debug_logger.info("Distributed Run: {}".format(False))
  debug_logger.info("cuDNN Enabled: {}".format(torch.backends.cudnn.enabled))
  debug_logger.info("cuDNN Benchmark: {}".format(torch.backends.cudnn.benchmark))

  model = load_model(hparams)

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

  train_loader, val_loader = prepare_dataloaders(hparams, trainset, valset)

  if not len(train_loader):
    debug_logger.error("Not enough trainingdata.")
    return False

  # Get shared output_directory ready
  # if rank == 0:
  #   if not os.path.isdir(output_directory):
  #     os.makedirs(output_directory)
  #     os.chmod(output_directory, 0o775)
  #   print("output directory", output_directory)

  #if hparams.with_tensorboard and rank == 0:

  model.train()

  total_its = hparams.epochs * len(train_loader)
  train_start = time.perf_counter()
  epoch_offset = max(0, int(iteration / len(train_loader)))
  # ================ MAIN TRAINING LOOP! ===================
  for epoch in range(epoch_offset, hparams.epochs):
    debug_logger.info("Epoch: {}".format(epoch))
    for i, batch in enumerate(train_loader):
      start = time.perf_counter()

      model.zero_grad()

      x = model.parse_batch(batch)
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

      duration = time.perf_counter() - start
      debug_logger.info("Epoch: {}/{} | Iteration: {}/{} | Total iteration: {}/{} | Train loss: {:.9f} | Duration: {:.2f}s/it | Total Duration: {:.2f}h".format(
        str(epoch).zfill(len(str(hparams.epochs))),
        hparams.epochs,
        str(i).zfill(len(str(len(train_loader) - 1))),
        len(train_loader) - 1,
        str(iteration).zfill(len(str(total_its))),
        total_its,
        reduced_loss,
        duration,
        (time.perf_counter() - train_start) / 60 / 60
      ))
      logger.log_training(reduced_loss, learning_rate, duration, iteration)

      #if hparams.with_tensorboard and rank == 0:
      logger.add_scalar('training_loss', reduced_loss, i + len(train_loader) * epoch)
      
      if (iteration % hparams.iters_per_checkpoint == 0):
        #if rank == 0:
        save_checkpoint(model, optimizer, hparams.learning_rate, iteration, save_checkpoint_dir)
        validate(model, criterion, val_loader, iteration, logger)

      iteration += 1

  debug_logger.info('Finished training.')
  duration_s = time.time() - complete_start
  duration_m = duration_s / 60
  debug_logger.info('Duration: {:.2f}min'.format(duration_m))


def train(custom_hparams: str, logdir: str, trainset: PreparedDataList, valset: PreparedDataList, save_checkpoint_dir: str, continue_train: bool):
  hp = create_hparams(custom_hparams)

  last_checkpoint_path = ""
  if continue_train:
    last_checkpoint_path, _ = get_last_checkpoint(save_checkpoint_dir)

  model, optimizer, learning_rate, iteration = _train(last_checkpoint_path, hp)
  train_core(hp, logdir, trainset, valset, save_checkpoint_dir, iteration, model, optimizer, learning_rate)

def _train(checkpoint_path: str, hparams):
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

  torch.backends.cudnn.enabled = True
  torch.backends.cudnn.benchmark = False

  torch.manual_seed(hparams.seed)
  torch.cuda.manual_seed(hparams.seed)

  # rank = 0 # 'rank of current gpu'
  # n_gpus = torch.cuda.device_count() # 'number of gpus'
  # if n_gpus > 1:
  #   raise Exception("More than one GPU is currently not supported.")
  #group_name = "group_name" # 'Distributed group name'
  # if n_gpus > 1:
  #   if args.group_name == '':
  #     print("WARNING: Multiple GPUs detected but no distributed group set")
  #     print("Only running 1 GPU.  Use distributed.py for multiple GPUs")
  #     n_gpus = 1
  # if n_gpus == 1 and args.rank != 0:
  #   raise Exception("Doing single GPU training on rank > 0")

  model = load_model(hparams)
  learning_rate = hparams.learning_rate
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

  # if hparams.fp16_run:
  #   from apex import amp
  #   model, optimizer = amp.initialize(model, optimizer, opt_level='O2')

  # if hparams.distributed_run:
  #   model = apply_gradient_allreduce(model)

  # Load checkpoint if one exists
  iteration = 0
  if checkpoint_path:
    model, optimizer, _learning_rate, iteration = load_checkpoint(checkpoint_path, model, optimizer)

    # if hparams.use_saved_learning_rate:
    #   learning_rate = _learning_rate
    iteration += 1  # next iteration is iteration + 1
  else:
    debug_logger.info("Starting new model...")
  
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
  filepath = os.path.join(parent_dir, get_pytorch_filename(iteration))
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
