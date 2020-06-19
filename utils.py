import numpy as np
import pandas as pd
from scipy.io.wavfile import read
import torch
from ipa2symb import extract_from_sentence
from collections import OrderedDict

csv_separator = '\t'

def parse_map(path: str) -> OrderedDict:
  with open(path, 'r') as f:
    tmp = f.readlines()
  #ipa_mapping = {x.strip()[0]: x.strip()[-1] for x in tmp}
  ipa_mapping = OrderedDict()
  for x in tmp:
    if '->' in x:
      from_to = x.rstrip('\n').replace(' -> ', '')
      symbs = extract_from_sentence(from_to, ignore_tones=False, ignore_arcs=False)
      a = symbs[0]
      if len(symbs) > 2:
        b = symbs[1:]
      else:
        b = [symbs[1]]
    else:
      a = x.rstrip('\n')
      b = ''
    ipa_mapping[a] = b
  return ipa_mapping

def get_mask_from_lengths(lengths):
  max_len = torch.max(lengths).item()
  ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
  mask = (ids < lengths.unsqueeze(1)).bool()
  return mask


def load_wav_to_torch(full_path):
  sampling_rate, data = read(full_path)
  return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_symbols(filename):
  data = pd.read_csv(filename, header=None, sep=csv_separator)
  #wavpath_col = 0
  #symbols_str_col = 1
  #data = data.iloc[:,[wavpath_col, symbols_str_col]]
  return data.values


def to_gpu(x):
  x = x.contiguous()

  if torch.cuda.is_available():
    x = x.cuda(non_blocking=True)
  return torch.autograd.Variable(x)
