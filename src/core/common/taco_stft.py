from typing import Dict, Optional

import numpy as np
import torch
from librosa.filters import mel as librosa_mel_fn

from src.core.common.audio import wav_to_float32_tensor
from src.core.common.stft import STFT
from src.core.common.train import overwrite_custom_hparams
from src.core.tacotron.hparams import AudioHParams


def dynamic_range_compression(x, C=1, clip_val=1e-5):
  """
  PARAMS
  ------
  C: compression factor
  """
  return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression(x, C=1):
  """
  PARAMS
  ------
  C: compression factor used to compress
  """
  return torch.exp(x) / C


def get_mel(wav_path: str, custom_hparams: Optional[Dict[str, str]]) -> np.ndarray:
  hparams = AudioHParams()
  hparams = overwrite_custom_hparams(hparams, custom_hparams)
  taco_stft = TacotronSTFT.fromhparams(hparams)
  orig_mel = taco_stft.get_mel_tensor_from_file(wav_path).numpy()
  return orig_mel


class TacotronSTFT(torch.nn.Module):
  def __init__(self, filter_length=1024, hop_length=256, win_length=1024,
               n_mel_channels=80, sampling_rate=22050, mel_fmin=0.0,
               mel_fmax=8000.0):
    super(TacotronSTFT, self).__init__()
    self.n_mel_channels = n_mel_channels
    self.sampling_rate = sampling_rate
    self.stft_fn = STFT(filter_length, hop_length, win_length)
    mel_basis = librosa_mel_fn(sampling_rate, filter_length, n_mel_channels, mel_fmin, mel_fmax)
    mel_basis = torch.from_numpy(mel_basis).float()
    self.register_buffer('mel_basis', mel_basis)

  @classmethod
  def fromhparams(cls, hparams: AudioHParams):
    return cls(
      filter_length=hparams.filter_length,
      hop_length=hparams.hop_length,
      win_length=hparams.win_length,
      n_mel_channels=hparams.n_mel_channels,
      sampling_rate=hparams.sampling_rate,
      mel_fmin=hparams.mel_fmin,
      mel_fmax=hparams.mel_fmax
    )

  def spectral_normalize(self, magnitudes):
    output = dynamic_range_compression(magnitudes)
    return output

  def spectral_de_normalize(self, magnitudes):
    output = dynamic_range_decompression(magnitudes)
    return output

  def mel_spectrogram(self, y):
    """Computes mel-spectrograms from a batch of waves
    PARAMS
    ------
    y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

    RETURNS
    -------
    mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
    """
    assert(torch.min(y.data) >= -1)
    assert(torch.max(y.data) <= 1)

    magnitudes, phases = self.stft_fn.transform(y)
    magnitudes = magnitudes.data
    mel_output = torch.matmul(self.mel_basis, magnitudes)
    mel_output = self.spectral_normalize(mel_output)
    return mel_output

  def get_mel_tensor_from_file(self, wav_path: str) -> torch.Tensor:
    wav_tensor, sr = wav_to_float32_tensor(wav_path)

    if sr != self.sampling_rate:
      raise ValueError(f"{wav_path} {sr} SR doesn't match target {self.sampling_rate} SR")

    return self.get_mel_tensor(wav_tensor)

  def get_mel_tensor(self, wav_tensor: torch.Tensor) -> torch.Tensor:
    wav_tensor = wav_tensor.unsqueeze(0)
    wav_tensor = torch.autograd.Variable(wav_tensor, requires_grad=False)
    melspec = self.mel_spectrogram(wav_tensor)
    melspec = melspec.squeeze(0)
    return melspec
