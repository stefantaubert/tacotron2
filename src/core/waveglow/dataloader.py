import random
from logging import Logger
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.core.common.audio import get_wav_tensor_segment
from src.core.common.taco_stft import TacotronSTFT
from src.core.pre.merge_ds import PreparedDataList
from src.core.waveglow.hparams import HParams


class MelLoader(Dataset):
  """
  This is the main class that calculates the spectrogram and returns the
  spectrogram, audio pair.
  """

  def __init__(self, prepare_ds_data: PreparedDataList, hparams: HParams, logger: Logger):
    self.taco_stft = TacotronSTFT(hparams, logger=logger)
    self.hparams = hparams
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
        cache[i] = self.taco_stft.get_wav_tensor_from_file(wav_path)
      self._logger.info("Done")
      self.cache = cache

  def __getitem__(self, index):
    if self.hparams.cache_wavs:
      wav_tensor = self.cache[index].clone().detach()
    else:
      wav_tensor = self.taco_stft.get_wav_tensor_from_file(self.wav_paths[index])
    wav_tensor = get_wav_tensor_segment(wav_tensor, self.hparams.segment_length)
    mel_tensor = self.taco_stft.get_mel_tensor(wav_tensor)
    return (mel_tensor, wav_tensor)

  def __len__(self):
    return len(self.wav_paths)


def parse_batch(batch) -> Tuple[torch.autograd.Variable, torch.autograd.Variable]:
  mel, audio = batch
  mel = torch.autograd.Variable(mel.cuda())
  audio = torch.autograd.Variable(audio.cuda())
  return (mel, audio), (mel, audio)


def prepare_trainloader(hparams: HParams, trainset: PreparedDataList, logger: Logger):
  logger.info(
    f"Duration trainset {trainset.get_total_duration_s() / 60:.2f}min / {trainset.get_total_duration_s() / 60 / 60:.2f}h")

  trn = MelLoader(trainset, hparams, logger)

  train_sampler = None
  shuffle = False  # maybe set to true bc taco is also true

  train_loader = DataLoader(
    dataset=trn,
    num_workers=0,
    shuffle=shuffle,
    sampler=train_sampler,
    batch_size=hparams.batch_size,
    pin_memory=False,
    drop_last=True
  )

  return train_loader


def prepare_valloader(hparams: HParams, valset: PreparedDataList, logger: Logger):
  logger.info(
    f"Duration valset {valset.get_total_duration_s() / 60:.2f}min / {valset.get_total_duration_s() / 60 / 60:.2f}h")

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

  return val_loader
