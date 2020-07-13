import matplotlib.pylab as plt
import torch
import numpy as np
import sys
import random
import os

from layers import TacotronSTFT
from utils import load_filepaths_and_symbols
from scipy.io.wavfile import read
from hparams import create_hparams
from tqdm import tqdm

def load_wav_to_torch(full_path):
  """
  Loads wavdata into torch array
  """
  sampling_rate, data = read(full_path)
  return torch.from_numpy(data).float(), sampling_rate

MAX_WAV_VALUE = 32768.0

def get_segment(audio, segment_length=16000):
  if audio.size(0) >= segment_length:
    max_audio_start = audio.size(0) - segment_length
    audio_start = random.randint(0, max_audio_start)
    audio = audio[audio_start:audio_start+segment_length]
  else:
    audio = torch.nn.functional.pad(audio, (0, segment_length - audio.size(0)), 'constant').data
  return audio

def get_audio(filename, target_sr=22050):
  # Read audio
  audio, sampling_rate = load_wav_to_torch(filename)
  if sampling_rate != target_sr:
    raise ValueError("{} SR doesn't match target {} SR".format(sampling_rate, target_sr))
  return audio

class Mel2Samp():
  """
  This is the main class that calculates the spectrogram and returns the
  spectrogram, audio pair.
  """
  def __init__(self, hparams):
    random.seed(1234)
    self.stft = TacotronSTFT(
      filter_length=hparams.filter_length,
      hop_length=hparams.hop_length,
      win_length=hparams.win_length,
      sampling_rate=hparams.sampling_rate,
      mel_fmin=hparams.mel_fmin,
      mel_fmax=hparams.mel_fmax
    )

  def get_mel(self, audio):
    audio_norm = audio / MAX_WAV_VALUE
    audio_norm = audio_norm.unsqueeze(0)
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
    melspec = self.stft.mel_spectrogram(audio_norm)
    melspec = torch.squeeze(melspec, 0)
    return melspec

def plot_melspecs(melspecs: list, mel_dim_x = 16, mel_dim_y = 5, factor=1) -> None:
  fig, axes = plt.subplots(len(melspecs), 1, figsize=(mel_dim_x*factor, len(melspecs)*mel_dim_y*factor))
  for i, mel in enumerate(melspecs):
    axes[i].imshow(mel, aspect='auto', origin='bottom', interpolation='none')


if __name__ == "__main__":
  hparams = create_hparams()

  mel2samp = Mel2Samp(hparams)
  
  wav_paths = [
    "/datasets/thchs_16bit_22050kHz/wav/train/C18/C18_742.wav",
    "/datasets/thchs_16bit_22050kHz/wav/train/C18/C18_743.wav",
    "/datasets/thchs_16bit_22050kHz/wav/train/C18/C18_744.wav"
  ]

  melspectrograms = []
  for w in tqdm(wav_paths):
    audio = get_audio(w, target_sr=hparams.sampling_rate)
    audio = get_segment(audio)
    melspectrogram = mel2samp.get_mel(audio)
    #new_filepath = "/tmp/out.pt"
    #torch.save(melspectrogram, new_filepath)
    melspectrograms.append(melspectrogram)

  plot_melspecs(melspectrograms, mel_dim_x = 4, factor=1)
  plt.savefig("/tmp/plot.png", bbox_inches='tight')
  plt.show()
