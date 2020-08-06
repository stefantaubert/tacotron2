import json
import os
import tarfile
from collections import OrderedDict
from dataclasses import astuple
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import wget
from scipy.io.wavfile import read
from skimage.metrics import structural_similarity
from pathlib import Path
from PIL import Image

import torch
from src.text.ipa2symb import extract_from_sentence

__csv_separator = '\t'

def split_train_test_val(wholeset: list, test_size: float, validation_size: float, seed: int) -> (list, list, list):
  trainset, testset, valset = wholeset, [], []

  if test_size > 0:
    trainset, testset = train_test_split(trainset, test_size=test_size, random_state=seed)

  if validation_size > 0:
    trainset, valset = train_test_split(trainset, test_size=validation_size, random_state=seed)
  
  return trainset, testset, valset

def stack_images_vertically(list_im, out_path):
  images = [Image.open(i) for i in list_im]
  widths, heights = zip(*(i.size for i in images))

  total_height = sum(heights)
  max_width = max(widths)

  new_im = Image.new(
    mode='RGB',
    size=(max_width, total_height),
    color=(255, 255, 255) # white
  )

  y_offset = 0
  for im in images:
    new_im.paste(im, (0, y_offset))
    y_offset += im.size[1]
  new_im.save(out_path)

def save_csv(data: list, path: str):
  data = [astuple(xi) for xi in data]
  df = pd.DataFrame(data)
  df.to_csv(path, header=None, index=None, sep=__csv_separator)

def load_csv(path: str, dc_type) -> list:
  data = pd.read_csv(path, header=None, sep=__csv_separator)
  data_loaded = [dc_type(*xi) for xi in data.values]
  return data_loaded

def get_last_checkpoint(checkpoint_dir) -> str:
  #checkpoint_dir = get_checkpoint_dir(training_dir_path)
  _, _, filenames = next(os.walk(checkpoint_dir))
  filenames = [x for x in filenames if ".log" not in x]
  at_least_one_checkpoint_exists = len(filenames) > 0
  if at_least_one_checkpoint_exists:
    last_checkpoint = str(max(list(map(int, filenames))))
    return last_checkpoint
  else:
    return None

def create_parent_folder(file: str):
  path = Path(file)
  os.makedirs(path.parent, exist_ok=True)
  return path.parent

def str_to_int(val: str) -> int:
  '''maps a string to int'''
  mapped = [(i + 1) * ord(x) for i, x in enumerate(val)]
  res = sum(mapped)
  return res

def get_subdir(training_dir_path: str, subdir: str, create: bool = True) -> str:
  result = os.path.join(training_dir_path, subdir)
  if create:
    os.makedirs(result, exist_ok=True)
  return result

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

def to_gpu(x):
  x = x.contiguous()

  if torch.cuda.is_available():
    x = x.cuda(non_blocking=True)
  return torch.autograd.Variable(x)

if __name__ == "__main__":
  x = "/datasets/models/taco2pt_v2/debug/filelist/filelist.csv"
  data = pd.read_csv(x, header=None, sep=__csv_separator)
  #get_speaker_count_csv(data)
