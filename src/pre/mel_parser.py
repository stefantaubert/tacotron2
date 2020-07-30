import os
import random

import tensorflow as tf
from tqdm import tqdm

import torch
from src.common.audio.utils import wav_to_float32
from src.paths import get_mels_dir, mels_file_name
from src.tacotron.layers import TacotronSTFT

def __create_hparams(hparams_string=None, verbose=False):
  """Create model hyperparameters. Parse nondefault from given string."""

  hparams = tf.contrib.training.HParams(
    #segment_length=16000, # waveglow 16000, tacotron None
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
  def __init__(self, custom_hparams: str):
    super().__init__()
    hparams = __create_hparams(custom_hparams)
    self.stft = TacotronSTFT(
      filter_length=hparams.filter_length,
      hop_length=hparams.hop_length,
      win_length=hparams.win_length,
      n_mel_channels=hparams.n_mel_channels,
      sampling_rate=hparams.sampling_rate,
      mel_fmin=hparams.mel_fmin,
      mel_fmax=hparams.mel_fmax
    )

  def get_mel(self, path: str, segment_length: int = 0) -> tuple:
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
      wav_tensor = __get_segment(wav_tensor, segment_length)
    
    mel = self.__get_mel_core(wav_tensor)

    return (mel, wav_tensor, duration)

  def __get_mel_core(self, wav_tensor):
    wav_tensor = wav_tensor.unsqueeze(0)
    wav_tensor = torch.autograd.Variable(wav_tensor, requires_grad=False)
    melspec = self.stft.mel_spectrogram(wav_tensor)
    melspec = melspec.squeeze(0)
    return melspec

def __get_segment(wav_tensor, segment_length: int):
  if wav_tensor.size(0) >= segment_length:
    max_audio_start = wav_tensor.size(0) - segment_length
    audio_start = random.randint(0, max_audio_start)
    wav_tensor = wav_tensor[audio_start:audio_start+segment_length]
  else:
    fill_size = segment_length - wav_tensor.size(0)
    wav_tensor = torch.nn.functional.pad(wav_tensor, (0, fill_size), 'constant').data
  
  return wav_tensor

if __name__ == "__main__":
  wav_path = "/datasets/thchs_16bit_22050kHz_nosil/wav/train/A32/A32_11.wav"
  mel_parser = MelParser()
  mel, mel_tensor, duration = mel_parser.get_mel(wav_path, segment_length=4000)
  print(mel[:,:8])
  print(mel.size())
  print(mel_tensor.size())

  mel, mel_tensor, duration = mel_parser.get_mel(wav_path, segment_length=2000)
  print(mel[:,:8])
  print(mel.size())
  print(mel_tensor.size())
