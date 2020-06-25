import numpy as np
import pandas as pd
from scipy.io.wavfile import read
import torch
import json
from ipa2symb import extract_from_sentence
from collections import OrderedDict

csv_separator = '\t'

wavpath_col = 1
symbols_str_col = 2
speaker_id_col = 4


def parse_ds_speakers(ds_speakers: str):
  speakers = ds_speakers.split(';')
  ds_speakers = [x.split(',') + [i] for i, x in enumerate(speakers)]
  return ds_speakers

def args_to_str(args):
  res = ""
  for arg, value in sorted(vars(args).items()):
    res += "{}: {}\n".format(arg, value)
  return res

def parse_map_json(path: str) -> dict:
  with open(path, 'r') as f:
    tmp = json.load(f)
  return tmp

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
  data = data.iloc[:, [wavpath_col, symbols_str_col, speaker_id_col]]
  data = data.values
  return data


def to_gpu(x):
  x = x.contiguous()

  if torch.cuda.is_available():
    x = x.cuda(non_blocking=True)
  return torch.autograd.Variable(x)
