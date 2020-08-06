import imageio
import matplotlib.pylab as plt
import numpy as np
from librosa.filters import mel as librosa_mel_fn
from skimage.metrics import structural_similarity

import torch
from src.common.audio.utils import wav_to_float32_tensor
from src.common.stft import STFT

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

  def get_mel_tensor_from_file(self, wav_path: str) -> torch.float32:
    wav_tensor, sr = wav_to_float32_tensor(wav_path)
    
    if sr != self.stft.sampling_rate :
      raise ValueError("{} {} SR doesn't match target {} SR".format(wav_path, sr, self.stft.sampling_rate))
    
    return self.get_mel_tensor(wav_tensor)

  def get_mel_tensor(self, wav_tensor: torch.float32) -> torch.float32:
    wav_tensor = wav_tensor.unsqueeze(0)
    wav_tensor = torch.autograd.Variable(wav_tensor, requires_grad=False)
    melspec = self.stft.mel_spectrogram(wav_tensor)
    melspec = melspec.squeeze(0)
    return melspec

def compare_mels(path_a, path_b):
  img_a = imageio.imread(path_a)
  img_b = imageio.imread(path_b)
  #img_b = imageio.imread(path_original_plot)
  assert img_a.shape[0] == img_b.shape[0]
  img_a_width = img_a.shape[1]
  img_b_width = img_b.shape[1]
  resize_width = img_a_width if img_a_width < img_b_width else img_b_width
  img_a = img_a[:,:resize_width]
  img_b = img_b[:,:resize_width]
  #imageio.imsave("/tmp/a.png", img_a)
  #imageio.imsave("/tmp/b.png", img_b)
  score, diff_img = structural_similarity(img_a, img_b, full=True, multichannel=True)
  #imageio.imsave(path_out, diff)
  return score, diff_img

def plot_melspec(mel, mel_dim_x=16, mel_dim_y=5, factor=1, title=None):
  height, width = mel.shape
  width_factor = width / 1000
  fig, axes = plt.subplots(
    nrows=1,
    ncols=1,
    figsize=(mel_dim_x*factor*width_factor, mel_dim_y*factor),
  )

  axes.set_title(title)
  axes.set_yticks(np.arange(0, height, step=10))
  axes.set_xticks(np.arange(0, width, step=100))
  axes.set_xlabel("Samples")
  axes.set_ylabel("Freq. channel")
  axes.imshow(mel, aspect='auto', origin='bottom', interpolation='none')
  return axes
  

if __name__ == "__main__":
  from src.tacotron.hparams import create_hparams
  from src.common.audio.utils import get_wav_tensor_segment
  wav_path = "/datasets/thchs_16bit_22050kHz_nosil/wav/train/A32/A32_11.wav"
  hparams = create_hparams()
  mel_parser = MelParser(hparams)
  mel = mel_parser.get_mel_tensor_from_file(wav_path)
  print(mel[:,:8])
  print(mel.size())

  wav_tensor, _ = wav_to_float32_tensor(wav_path)
  wav_tensor = get_wav_tensor_segment(wav_tensor, segment_length=4000)
  mel = mel_parser.get_mel_tensor(wav_tensor)
  print(mel[:,:8])
  print(mel.size())
