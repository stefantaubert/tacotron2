from dataclasses import dataclass
from logging import Logger, getLogger
from typing import Dict, Optional

import numpy as np
import torch
from librosa.filters import mel as librosa_mel_fn

from src.core.common.audio import wav_to_float32_tensor
from src.core.common.stft import STFT
from src.core.common.train import overwrite_custom_hparams


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
  hparams = STFTHParams()
  hparams = overwrite_custom_hparams(hparams, custom_hparams)
  taco_stft = TacotronSTFT(hparams, logger=getLogger())
  orig_mel = taco_stft.get_mel_tensor_from_file(wav_path).numpy()
  return orig_mel


@dataclass
class STFTHParams():
  n_mel_channels: int = 80
  sampling_rate: int = 22050
  filter_length: int = 1024
  hop_length: int = 256
  win_length: int = 1024
  mel_fmin: float = 0.0
  mel_fmax: float = 8000.0


class TacotronSTFT(torch.nn.Module):
  def __init__(self, hparams: STFTHParams, logger: Logger):
    super(TacotronSTFT, self).__init__()
    self.logger = logger
    self.n_mel_channels = hparams.n_mel_channels
    self.sampling_rate = hparams.sampling_rate
    self.stft_fn = STFT(hparams.filter_length, hparams.hop_length, hparams.win_length)
    mel_basis = librosa_mel_fn(
      sr=hparams.sampling_rate,
      n_fft=hparams.filter_length,
      n_mels=hparams.n_mel_channels,
      fmin=hparams.mel_fmin,
      fmax=hparams.mel_fmax
    )
    mel_basis = torch.from_numpy(mel_basis).float()
    self.register_buffer('mel_basis', mel_basis)

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

  def get_wav_tensor_from_file(self, wav_path: str) -> torch.Tensor:
    wav_tensor, sampling_rate = wav_to_float32_tensor(wav_path)

    if sampling_rate != self.sampling_rate:
      msg = f"{wav_path}: The sampling rate of the file ({sampling_rate}Hz) doesn't match the target sampling rate ({self.sampling_rate}Hz)!"
      self.logger.exception(msg)
      raise ValueError(msg)

    return wav_tensor

  def get_mel_tensor_from_file(self, wav_path: str) -> torch.Tensor:
    wav_tensor = self.get_wav_tensor_from_file(wav_path)
    return self.get_mel_tensor(wav_tensor)

  def get_mel_tensor(self, wav_tensor: torch.Tensor) -> torch.Tensor:
    wav_tensor = wav_tensor.unsqueeze(0)
    wav_tensor = torch.autograd.Variable(wav_tensor, requires_grad=False)
    melspec = self.mel_spectrogram(wav_tensor)
    melspec = melspec.squeeze(0)
    return melspec
