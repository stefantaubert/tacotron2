import numpy as np
import pandas as pd
from scipy.io.wavfile import read
import torch
import json
from ipa2symb import extract_from_sentence
from collections import OrderedDict
import wget
import os
import tarfile

csv_separator = '\t'

utt_name_col = 0
wavpath_col = 1
symbols_str_col = 2
duration_col = 3
speaker_id_col = 4
speaker_name_col = 5


def download_tar(download_url, dir_path, tarmode: str = "r:gz"):
  print("Starting download of {}...".format(download_url))
  os.makedirs(dir_path, exist_ok=True)
  dest = wget.download(download_url, dir_path)
  downloaded_file = os.path.join(dir_path, dest)
  print("\nFinished download to {}".format(downloaded_file))
  print("Unpacking...")
  tar = tarfile.open(downloaded_file, tarmode)
  tar.extractall(dir_path)
  tar.close()
  os.remove(downloaded_file)
  print("Done.")

def get_utterance_names_csv(csv) -> list:
  all_names = set(np.unique(csv.iloc[:, [utt_name_col]].values))
  return all_names

def get_speakers_csv(csv) -> set:
  all_speakers = set(np.unique(csv.iloc[:, [speaker_id_col]].values))
  return all_speakers

def get_speaker_count_csv(csv) -> int:
  speaker_count = len(get_speakers_csv(csv))
  return speaker_count

def get_total_duration_min_df(csv_file) -> float:
  data = pd.read_csv(csv_file, header=None, sep=csv_separator)
  return get_total_duration_min(data)

def get_total_duration_min(dataset_csv) -> float:
  total_dur_min = float(dataset_csv.iloc[:, [duration_col]].sum(axis=0)) / 60
  return total_dur_min

def serialize_ds_speaker(ds: str, speaker: str):
  return "{},{}".format(ds, speaker)

def parse_ds_speaker(ds_speaker: str):
  return ds_speaker.split(',')

def serialize_ds_speakers(ds_speakers: tuple):
  ds_speakers = [serialize_ds_speaker(ds, speaker) for ds, speaker in ds_speakers]
  res = ";".join(ds_speakers)
  return res

def parse_ds_speakers(ds_speakers: str) -> list:
  '''
  Example: [ ['thchs', 'C11', 0], ... ]
  '''
  speakers = ds_speakers.split(';')
  ds_speakers = [parse_ds_speaker(x) + [i] for i, x in enumerate(speakers)]
  return ds_speakers

def args_to_str(args):
  res = ""
  for arg, value in sorted(vars(args).items()):
    res += "{}: {}\n".format(arg, value)
  return res

def parse_json(path: str) -> dict:
  with open(path, 'r', encoding='utf-8') as f:
    tmp = json.load(f)
  return tmp

def save_json(path: str, mapping_dict: dict):
  with open(path, 'w', encoding='utf-8') as f:
    json.dump(mapping_dict, f, ensure_ascii=False, indent=2)

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

if __name__ == "__main__":
  x = "/datasets/models/taco2pt_v2/debug/filelist/filelist.csv"
  data = pd.read_csv(x, header=None, sep=csv_separator)
  #get_speaker_count_csv(data)
  get_utterance_names_csv(data)
