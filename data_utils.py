import random
import numpy as np
import torch
import torch.utils.data

import layers
from utils import load_wav_to_torch, load_filepaths_and_symbols
from text.conversion.SymbolConverter import get_symbols_from_str


class SymbolsMelLoader(torch.utils.data.Dataset):
  """
    1) loads audio,text pairs
    2) normalizes text and converts them to sequences of one-hot vectors
    3) computes mel-spectrograms from audio files.
  """
  def __init__(self, audiopaths_and_text, hparams):
    self.audiopaths_and_symbols = load_filepaths_and_symbols(audiopaths_and_text)
    self.max_wav_value = hparams.max_wav_value
    self.sampling_rate = hparams.sampling_rate
    self.load_mel_from_disk = hparams.load_mel_from_disk
    self.stft = layers.TacotronSTFT(
      hparams.filter_length, hparams.hop_length, hparams.win_length,
      hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
      hparams.mel_fmax)
    random.seed(hparams.seed)
    random.shuffle(self.audiopaths_and_symbols)

  def get_mel_symbols_pair(self, audiopath_and_text):
    # separate filename and text
    audiopath, symbols_str, speaker_id = audiopath_and_text[0], audiopath_and_text[1], audiopath_and_text[2]
    symbols_tensor = self.get_symbols(symbols_str)
    mel_tensor = self.get_mel(audiopath)
    return (symbols_tensor, mel_tensor, speaker_id)

  def get_mel(self, filename):
    if not self.load_mel_from_disk:
      audio, sampling_rate = load_wav_to_torch(filename)
      if sampling_rate != self.stft.sampling_rate:
        raise ValueError("{} {} SR doesn't match target {} SR".format(
          sampling_rate, self.stft.sampling_rate))
      audio_norm = audio / self.max_wav_value
      audio_norm = audio_norm.unsqueeze(0)
      audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
      melspec = self.stft.mel_spectrogram(audio_norm)
      melspec = torch.squeeze(melspec, 0)
    else:
      melspec = torch.from_numpy(np.load(filename))
      assert melspec.size(0) == self.stft.n_mel_channels, (
        'Mel dimension mismatch: given {}, expected {}'.format(
          melspec.size(0), self.stft.n_mel_channels))

    return melspec

  def get_symbols(self, symbols_str):
    symbols = get_symbols_from_str(symbols_str)
    symbols_tensor = torch.IntTensor(symbols)
    return symbols_tensor


  def __getitem__(self, index):
    return self.get_mel_symbols_pair(self.audiopaths_and_symbols[index])

  def __len__(self):
    return len(self.audiopaths_and_symbols)


class SymbolsMelCollate():
  """ Zero-pads model inputs and targets based on number of frames per step
  """
  def __init__(self, n_frames_per_step):
    self.n_frames_per_step = n_frames_per_step

  def __call__(self, batch):
    """Collate's training batch from normalized text and mel-spectrogram
    PARAMS
    ------
    batch: [text_normalized, mel_normalized]
    """
    # Right zero-pad all one-hot text sequences to max input length
    input_lengths, ids_sorted_decreasing = torch.sort(torch.LongTensor([len(x[0]) for x in batch]), dim=0, descending=True)
    max_input_len = input_lengths[0]

    text_padded = torch.LongTensor(len(batch), max_input_len)
    text_padded.zero_()
    for i in range(len(ids_sorted_decreasing)):
      text = batch[ids_sorted_decreasing[i]][0]
      text_padded[i, :text.size(0)] = text

    # Right zero-pad mel-spec
    num_mels = batch[0][1].size(0)
    max_target_len = max([x[1].size(1) for x in batch])
    if max_target_len % self.n_frames_per_step != 0:
      max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
      assert max_target_len % self.n_frames_per_step == 0

    # include mel padded and gate padded
    mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
    mel_padded.zero_()
    gate_padded = torch.FloatTensor(len(batch), max_target_len)
    gate_padded.zero_()
    output_lengths = torch.LongTensor(len(batch))
    for i in range(len(ids_sorted_decreasing)):
      mel = batch[ids_sorted_decreasing[i]][1]
      mel_padded[i, :, :mel.size(1)] = mel
      gate_padded[i, mel.size(1)-1:] = 1
      output_lengths[i] = mel.size(1)

    # count number of items - characters in text
    #len_x = []
    speaker_ids = []
    for i in range(len(ids_sorted_decreasing)):
      #len_symb = batch[ids_sorted_decreasing[i]][0].get_shape()[0]
      #len_x.append(len_symb)
      speaker_ids.append(batch[ids_sorted_decreasing[i]][2])

    #len_x = torch.Tensor(len_x)
    speaker_ids = torch.Tensor(speaker_ids)

    return text_padded, input_lengths, mel_padded, gate_padded, \
      output_lengths, speaker_ids

def batch_to_gpu(batch):
  text_padded, input_lengths, mel_padded, gate_padded, \
      output_lengths, len_x, speaker_ids = batch
  text_padded = to_gpu(text_padded).long()
  input_lengths = to_gpu(input_lengths).long()
  max_len = torch.max(input_lengths.data).item()
  mel_padded = to_gpu(mel_padded).float()
  gate_padded = to_gpu(gate_padded).float()
  output_lengths = to_gpu(output_lengths).long()
  speaker_ids = to_gpu(speaker_ids).long()
  x = (text_padded, input_lengths, mel_padded, max_len, output_lengths, speaker_ids)
  y = (mel_padded, gate_padded)
  len_x = torch.sum(output_lengths)

  return x, y, len_x

def to_gpu(x):
  x = x.contiguous()

  if torch.cuda.is_available():
    x = x.cuda(non_blocking=True)
  return torch.autograd.Variable(x)
