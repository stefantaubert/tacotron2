import random

import numpy as np
from tqdm import tqdm

import torch
import torch.utils.data
from src.tacotron.prepare_ds_ms_io import get_mel_path, get_serialized_ids, get_speaker_id
from src.tacotron.layers import TacotronSTFT
from src.text.symbol_converter import deserialize_symbol_ids


class SymbolsMelLoader(torch.utils.data.Dataset):
  """
    1) loads audio,text pairs
    2) normalizes text and converts them to sequences of one-hot vectors
    3) computes mel-spectrograms from audio files.
  """
  def __init__(self, prepare_ds_ms_data, hparams):
    data = prepare_ds_ms_data

    random.seed(hparams.seed)
    random.shuffle(data)

    print("Loading mels into memory...")
    self.cache = {}
    for i, values in tqdm(enumerate(data), total=len(data)):
      symbol_ids = deserialize_symbol_ids(get_serialized_ids(values))
      symbols_tensor = torch.IntTensor(symbol_ids)
      mel_tensor = torch.load(get_mel_path(values), map_location='cpu')
      self.cache[i] = (symbols_tensor, mel_tensor, get_speaker_id(values))

  def __getitem__(self, index):
    #return self.cache[index]
    symbols_tensor, mel_tensor, speaker_id = self.cache[index]
    return symbols_tensor.clone().detach(), mel_tensor.clone().detach(), speaker_id

  def __len__(self):
    return len(self.cache)


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
