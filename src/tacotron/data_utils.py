import random

import numpy as np

import torch
import torch.utils.data
from src.common.utils import load_filepaths_and_symbols, load_wav_to_torch
from src.tacotron.layers import TacotronSTFT
from src.text.symbol_converter import deserialize_symbol_ids
from src.common.audio.utils import wav_to_float32
from src.waveglow.mel2samp import MelParser

# def get_mel_(audio_path, stft):
#   from scipy.io.wavfile import read
#   sampling_rate, data = read(audio_path)
#   data = data.astype(np.float32)
#   audio = torch.FloatTensor(data)

#   #audio, sampling_rate = load_wav_to_torch(audio_path)
#   # if sampling_rate != self.stft.sampling_rate:
#   #   raise ValueError("{} {} SR doesn't match target {} SR".format(
#   #     sampling_rate, self.stft.sampling_rate))
#   audio_norm = audio / 32768.0
#   audio_norm = audio_norm.unsqueeze(0)
#   audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
#   melspec = stft.mel_spectrogram(audio_norm)
#   melspec = torch.squeeze(melspec, 0)
#   return melspec

# def get_mel(wav_tensor, stft):
#   wav_tensor = wav_tensor.unsqueeze(0)
#   wav_tensor = torch.autograd.Variable(wav_tensor, requires_grad=False)
#   melspec = stft.mel_spectrogram(wav_tensor)
#   melspec = torch.squeeze(melspec, 0)
#   return melspec

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
    self.mel_parser = MelParser(hparams)
    #self.load_mel_from_disk = hparams.load_mel_from_disk
    # self.stft = TacotronSTFT(
    #   filter_length=hparams.filter_length,
    #   hop_length=hparams.hop_length,
    #   win_length=hparams.win_length,
    #   n_mel_channels=hparams.n_mel_channels,
    #   sampling_rate=hparams.sampling_rate,
    #   mel_fmin=hparams.mel_fmin,
    #   mel_fmax=hparams.mel_fmax
    # )

    random.seed(hparams.seed)
    random.shuffle(self.audiopaths_and_symbols)

  def get_mel_symbols_pair(self, audiopath_and_text):
    # separate filename and text
    audiopath, serialized_symbol_ids, speaker_id = audiopath_and_text[0], audiopath_and_text[1], audiopath_and_text[2]
    symbols_tensor = self.get_symbols(serialized_symbol_ids)
    mel_tensor, _ = self.mel_parser.get_mel(audiopath, segment_length=None)
    #mel_tensor = self.get_mel(audiopath)

    # wav, sampling_rate = wav_to_float32(audiopath)

    # if sampling_rate != self.sampling_rate:
    #   raise ValueError("{} {} SR doesn't match target {} SR".format(audiopath, sampling_rate, self.sampling_rate))

    # wav_tensor = torch.FloatTensor(wav)
    # mel_tensor = get_mel(wav_tensor, self.stft)
    return (symbols_tensor, mel_tensor, speaker_id)

  # def get_mel(self, filename):
  #   if not self.load_mel_from_disk:
  #     audio, sampling_rate = load_wav_to_torch(filename)
  #     if sampling_rate != self.stft.sampling_rate:
  #       raise ValueError("{} {} SR doesn't match target {} SR".format(
  #         sampling_rate, self.stft.sampling_rate))
  #     audio_norm = audio / self.max_wav_value
  #     audio_norm = audio_norm.unsqueeze(0)
  #     audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
  #     melspec = self.stft.mel_spectrogram(audio_norm)
  #     melspec = torch.squeeze(melspec, 0)
  #   else:
  #     melspec = torch.from_numpy(np.load(filename))
  #     assert melspec.size(0) == self.stft.n_mel_channels, (
  #       'Mel dimension mismatch: given {}, expected {}'.format(
  #         melspec.size(0), self.stft.n_mel_channels))

  #   return melspec

  def get_symbols(self, serialized_symbol_ids):
    symbol_ids = deserialize_symbol_ids(serialized_symbol_ids)
    symbols_tensor = torch.IntTensor(symbol_ids)
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


    return text_padded, input_lengths, mel_padded, gate_padded, \
      output_lengths
