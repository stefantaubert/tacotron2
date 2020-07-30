import os
import random

import pandas as pd
import tensorflow as tf
from tqdm import tqdm

import torch
from src.common.audio.utils import wav_to_float32
from src.paths import get_mels_dir, mels_file_name
from src.tacotron.layers import TacotronSTFT

__csv_separator = "\t"


def __create_hparams(hparams_string=None, verbose=False):
  """Create model hyperparameters. Parse nondefault from given string."""

  hparams = tf.contrib.training.HParams(
    segment_length=16000, # waveglow 16000, tacotron None
    sampling_rate=22050,
    filter_length=1024,
    hop_length=256,
    win_length=1024,
    n_mel_channels=80,
    mel_fmin=0.0,
    mel_fmax=8000.0
  )

  if hparams_string:
    tf.compat.v1.logging.info('Parsing command line hparams: %s', hparams_string)
    hparams.parse(hparams_string)

  if verbose:
    tf.compat.v1.logging.info('Final parsed hparams: %s', hparams.values())

  return hparams


class MelParser():
  def __init__(self, hparams):
    super().__init__()
    self.stft = TacotronSTFT(
      filter_length=hparams.filter_length,
      hop_length=hparams.hop_length,
      win_length=hparams.win_length,
      n_mel_channels=hparams.n_mel_channels,
      sampling_rate=hparams.sampling_rate,
      mel_fmin=hparams.mel_fmin,
      mel_fmax=hparams.mel_fmax
    )

  def get_mel(self, path, segment_length=None) -> tuple:
    '''
    returns mel and wav_tensor
    - mel is float32 = FloatTensor
    '''
    wav, sampling_rate = wav_to_float32(path)
    duration = len(wav) / sampling_rate

    if sampling_rate != self.stft.sampling_rate:
      raise ValueError("{} {} SR doesn't match target {} SR".format(path, sampling_rate, self.stft.sampling_rate))
    
    wav_tensor = torch.FloatTensor(wav)

    if segment_length:
      # Take segment
      if wav_tensor.size(0) >= segment_length:
        max_audio_start = wav_tensor.size(0) - segment_length
        audio_start = random.randint(0, max_audio_start)
        wav_tensor = wav_tensor[audio_start:audio_start+segment_length]
      else:
        wav_tensor = torch.nn.functional.pad(wav_tensor, (0, segment_length - wav_tensor.size(0)), 'constant').data
    
    mel = self.__get_mel_core(wav_tensor)

    return (mel, wav_tensor, duration)
      
  def __get_mel_core(self, wav_tensor):
    wav_tensor = wav_tensor.unsqueeze(0)
    wav_tensor = torch.autograd.Variable(wav_tensor, requires_grad=False)
    melspec = self.stft.mel_spectrogram(wav_tensor)
    melspec = melspec.squeeze(0)
    return melspec

# output: basename, speaker_name, text, mel_path, duration
def parse_data(base_dir: str, name: str):
  dest_dir = get_mels_dir(base_dir, name)
  dest_file_path = os.path.join(dest_dir, mels_file_name)
  speaker_data = pd.read_csv(dest_file_path, header=None, sep=__csv_separator)
  speaker_data = speaker_data.values
  return speaker_data

def calc_mels(base_dir: str, name: str, data: list, custom_hparams: str):
  result = []
  hparams = __create_hparams(custom_hparams)
  mel_parser = MelParser(hparams)
  dest_dir = get_mels_dir(base_dir, name)
  # with torch.no_grad():
  for i, values in tqdm(enumerate(data), total=len(data)):
    name, speaker_name, text, wav_path = values[0], values[1], values[2], values[3]
    mel_path = os.path.join(dest_dir, "{}.pt".format(i))
    mel, _, duration = mel_parser.get_mel(wav_path, segment_length=hparams.segment_length)
    torch.save(mel, mel_path)
    result.append((name, speaker_name, text, mel_path, duration))
    
  dest_file_path = os.path.join(dest_dir, mels_file_name)
  df = pd.DataFrame(result)
  df.to_csv(dest_file_path, header=None, index=None, sep=__csv_separator)
  print("Dataset saved.")
