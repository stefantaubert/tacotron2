import random

import numpy as np
from tqdm import tqdm

import torch
import torch.utils.data
from src.tacotron.prepare_ds_ms_io import PreparedData, PreparedDataList
from src.tacotron.layers import TacotronSTFT
from src.text.symbol_converter import deserialize_symbol_ids


class SymbolsMelLoader(torch.utils.data.Dataset):
  """
    1) loads audio,text pairs
    2) normalizes text and converts them to sequences of one-hot vectors
    3) computes mel-spectrograms from audio files.
  """
  def __init__(self, prepare_ds_ms_data: PreparedDataList, hparams):
    data = prepare_ds_ms_data

    random.seed(hparams.seed)
    random.shuffle(data)
    
    print("Reading mels...")
    self.data = {}
    values: PreparedData
    for i, values in enumerate(tqdm(data)):
      symbol_ids = deserialize_symbol_ids(values.serialized_updated_ids)
      symbols_tensor = torch.IntTensor(symbol_ids)
      self.data[i] = (symbols_tensor, values.mel_path, values.speaker_id)
    
    if hparams.cache_mels:
      print("Loading mels into memory...")
      self.cache = {}
      for i, values in tqdm(self.data.items()):
        mel_tensor = torch.load(values[1], map_location='cpu')
        self.cache[i] = mel_tensor
    self.use_cache = hparams.cache_mels

  def __getitem__(self, index):
    #return self.cache[index]
    symbols_tensor, mel_path, speaker_id = self.data[index]
    if self.use_cache:
      mel_tensor = self.cache[index].clone().detach()
    else:
      mel_tensor = torch.load(mel_path, map_location='cpu')
    return symbols_tensor.clone().detach(), mel_tensor, speaker_id

  def __len__(self):
    return len(self.data)


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

    return text_padded, input_lengths, mel_padded, gate_padded, output_lengths, speaker_ids
