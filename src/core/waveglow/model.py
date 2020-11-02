# For copyright see LICENCE

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from src.core.waveglow.hparams import HParams


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
  n_channels_int = n_channels[0]
  in_act = input_a + input_b
  t_act = torch.tanh(in_act[:, :n_channels_int, :])
  s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
  acts = t_act * s_act
  return acts


class Invertible1x1Conv(torch.nn.Module):
  """
  The layer outputs both the convolution, and the log determinant
  of its weight matrix.  If reverse=True it does convolution with
  inverse
  """

  def __init__(self, c):
    super(Invertible1x1Conv, self).__init__()
    self.conv = torch.nn.Conv1d(c, c, kernel_size=1, stride=1, padding=0,
                                bias=False)

    # Sample a random orthonormal matrix to initialize weights
    W = torch.qr(torch.FloatTensor(c, c).normal_())[0]

    # Ensure determinant is 1.0 not -1.0
    if torch.det(W) < 0:
      W[:, 0] = -1 * W[:, 0]
    W = W.view(c, c, 1)
    self.conv.weight.data = W

  def forward(self, z, reverse=False):
    # shape
    batch_size, group_size, n_of_groups = z.size()

    W = self.conv.weight.squeeze()

    if reverse:
      if not hasattr(self, 'W_inverse'):
        # Reverse computation
        W_inverse = W.float().inverse()
        W_inverse = Variable(W_inverse[..., None])
        if z.type() == 'torch.cuda.HalfTensor':
          W_inverse = W_inverse.half()
        self.W_inverse = W_inverse
      z = F.conv1d(z, self.W_inverse, bias=None, stride=1, padding=0)
      return z
    else:
      # Forward computation
      log_det_W = batch_size * n_of_groups * torch.logdet(W)
      z = self.conv(z)
      return z, log_det_W


class WN(torch.nn.Module):
  """
  This is the WaveNet like layer for the affine coupling. The primary difference
  from WaveNet is the convolutions need not be causal. There is also no dilation
  size reset. The dilation only doubles on each layer
  """

  def __init__(self, n_in_channels, n_mel_channels, hparams: HParams):
    super(WN, self).__init__()
    assert(hparams.kernel_size % 2 == 1)
    assert(hparams.n_channels % 2 == 0)
    self.n_layers = hparams.n_layers
    self.n_channels = hparams.n_channels
    self.in_layers = torch.nn.ModuleList()
    self.res_skip_layers = torch.nn.ModuleList()

    start = torch.nn.Conv1d(n_in_channels, self.n_channels, 1)
    start = torch.nn.utils.weight_norm(start, name='weight')
    self.start = start

    # Initializing last layer to 0 makes the affine coupling layers
    # do nothing at first.  This helps with training stability
    end = torch.nn.Conv1d(self.n_channels, 2 * n_in_channels, 1)
    end.weight.data.zero_()
    end.bias.data.zero_()
    self.end = end

    cond_layer = torch.nn.Conv1d(n_mel_channels, 2 * self.n_channels * self.n_layers, 1)
    self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name='weight')

    for i in range(self.n_layers):
      dilation = 2 ** i
      padding = int((hparams.kernel_size * dilation - dilation) / 2)
      in_layer = torch.nn.Conv1d(self.n_channels, 2 * self.n_channels, hparams.kernel_size,
                                 dilation=dilation, padding=padding)
      in_layer = torch.nn.utils.weight_norm(in_layer, name='weight')
      self.in_layers.append(in_layer)

      # last one is not necessary
      if i < self.n_layers - 1:
        res_skip_channels = 2 * self.n_channels
      else:
        res_skip_channels = self.n_channels
      res_skip_layer = torch.nn.Conv1d(self.n_channels, res_skip_channels, 1)
      res_skip_layer = torch.nn.utils.weight_norm(res_skip_layer, name='weight')
      self.res_skip_layers.append(res_skip_layer)

  def forward(self, forward_input):
    audio, spect = forward_input
    audio = self.start(audio)
    output = torch.zeros_like(audio)
    n_channels_tensor = torch.IntTensor([self.n_channels])

    spect = self.cond_layer(spect)

    for i in range(self.n_layers):
      spect_offset = i * 2 * self.n_channels
      acts = fused_add_tanh_sigmoid_multiply(
        self.in_layers[i](audio),
        spect[:, spect_offset:spect_offset + 2 * self.n_channels, :],
        n_channels_tensor)

      res_skip_acts = self.res_skip_layers[i](acts)
      if i < self.n_layers - 1:
        audio = audio + res_skip_acts[:, :self.n_channels, :]
        output = output + res_skip_acts[:, self.n_channels:, :]
      else:
        output = output + res_skip_acts

    return self.end(output)


class WaveGlow(torch.nn.Module):
  def __init__(self, hparams: HParams):
    super().__init__()

    self.upsample = torch.nn.ConvTranspose1d(
      hparams.n_mel_channels,
      hparams.n_mel_channels,
      1024,
      stride=256
    )

    assert(hparams.n_group % 2 == 0)
    self.n_flows = hparams.n_flows
    self.n_group = hparams.n_group
    self.n_early_every = hparams.n_early_every
    self.n_early_size = hparams.n_early_size
    self.WN = torch.nn.ModuleList()
    self.convinv = torch.nn.ModuleList()

    n_half = int(self.n_group / 2)

    # Set up layers with the right sizes based on how many dimensions
    # have been output already
    n_remaining_channels = self.n_group
    for k in range(self.n_flows):
      if k % self.n_early_every == 0 and k > 0:
        n_half = n_half - int(self.n_early_size / 2)
        n_remaining_channels = n_remaining_channels - self.n_early_size
      self.convinv.append(Invertible1x1Conv(n_remaining_channels))
      WN_res = WN(
        n_in_channels=n_half,
        n_mel_channels=hparams.n_mel_channels * self.n_group,
        hparams=hparams
      )
      self.WN.append(WN_res)
    self.n_remaining_channels = n_remaining_channels  # Useful during inference

  def forward(self, forward_input):
    """
    forward_input[0] = mel_spectrogram:  batch x n_mel_channels x frames
    forward_input[1] = audio: batch x time
    """
    spect, audio = forward_input

    #  Upsample spectrogram to size of audio
    spect = self.upsample(spect)
    assert(spect.size(2) >= audio.size(1))
    if spect.size(2) > audio.size(1):
      spect = spect[:, :, :audio.size(1)]

    spect = spect.unfold(2, self.n_group, self.n_group).permute(0, 2, 1, 3)
    spect = spect.contiguous().view(spect.size(0), spect.size(1), -1).permute(0, 2, 1)

    audio = audio.unfold(1, self.n_group, self.n_group).permute(0, 2, 1)
    output_audio = []
    log_s_list = []
    log_det_W_list = []

    for k in range(self.n_flows):
      if k % self.n_early_every == 0 and k > 0:
        output_audio.append(audio[:, :self.n_early_size, :])
        audio = audio[:, self.n_early_size:, :]

      audio, log_det_W = self.convinv[k](audio)
      log_det_W_list.append(log_det_W)

      n_half = int(audio.size(1) / 2)
      audio_0 = audio[:, :n_half, :]
      audio_1 = audio[:, n_half:, :]

      output = self.WN[k]((audio_0, spect))
      log_s = output[:, n_half:, :]
      b = output[:, :n_half, :]
      audio_1 = torch.exp(log_s) * audio_1 + b
      log_s_list.append(log_s)

      audio = torch.cat([audio_0, audio_1], 1)

    output_audio.append(audio)
    return torch.cat(output_audio, 1), log_s_list, log_det_W_list

  def infer(self, spect, sigma=1.0):
    spect = self.upsample(spect)
    # trim conv artifacts. maybe pad spec to kernel multiple
    time_cutoff = self.upsample.kernel_size[0] - self.upsample.stride[0]
    spect = spect[:, :, :-time_cutoff]

    spect = spect.unfold(2, self.n_group, self.n_group).permute(0, 2, 1, 3)
    spect = spect.contiguous().view(spect.size(0), spect.size(1), -1).permute(0, 2, 1)

    if spect.type() == 'torch.cuda.HalfTensor':
      audio = torch.cuda.HalfTensor(spect.size(0),
                                    self.n_remaining_channels,
                                    spect.size(2)).normal_()
    else:
      audio = torch.cuda.FloatTensor(spect.size(0),
                                     self.n_remaining_channels,
                                     spect.size(2)).normal_()

    audio = torch.autograd.Variable(sigma * audio)

    for k in reversed(range(self.n_flows)):
      n_half = int(audio.size(1) / 2)
      audio_0 = audio[:, :n_half, :]
      audio_1 = audio[:, n_half:, :]

      output = self.WN[k]((audio_0, spect))

      s = output[:, n_half:, :]
      b = output[:, :n_half, :]
      audio_1 = (audio_1 - b) / torch.exp(s)
      audio = torch.cat([audio_0, audio_1], 1)

      audio = self.convinv[k](audio, reverse=True)

      if k % self.n_early_every == 0 and k > 0:
        if spect.type() == 'torch.cuda.HalfTensor':
          z = torch.cuda.HalfTensor(spect.size(0), self.n_early_size, spect.size(2)).normal_()
        else:
          z = torch.cuda.FloatTensor(spect.size(0), self.n_early_size, spect.size(2)).normal_()
        audio = torch.cat((sigma * z, audio), 1)

    audio = audio.permute(0, 2, 1).contiguous().view(audio.size(0), -1).data
    return audio

  @staticmethod
  def remove_weightnorm(model):
    # see: zotero://select/library/items/KIY65PZJ
    waveglow = model
    for wnet in waveglow.WN:
      wnet.start = torch.nn.utils.remove_weight_norm(wnet.start)
      wnet.in_layers = remove(wnet.in_layers)
      wnet.cond_layer = torch.nn.utils.remove_weight_norm(wnet.cond_layer)
      wnet.res_skip_layers = remove(wnet.res_skip_layers)
    return waveglow


def remove(conv_list):
  new_conv_list = torch.nn.ModuleList()
  for old_conv in conv_list:
    old_conv = torch.nn.utils.remove_weight_norm(old_conv)
    new_conv_list.append(old_conv)
  return new_conv_list
