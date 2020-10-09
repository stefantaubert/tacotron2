import json
import logging
import os
import random
import tarfile
import unicodedata
from collections import Counter
from dataclasses import astuple
from pathlib import Path
from typing import Any, Generic, List, Set, Tuple, Type, TypeVar, Union

import numpy as np
import pandas as pd
import torch
import wget
from matplotlib.figure import Figure
from PIL import Image
from scipy.spatial.distance import cosine
from tqdm import tqdm

from src.core.common.globals import CSV_SEPERATOR

T = TypeVar('T')


def pass_lines(method: Any, text: str):
  lines = text.split("\n")
  for l in lines:
    method(l)


def figure_to_numpy(fig: Figure):
  data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
  data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  return data


def get_filenames(parent_dir: str) -> List[str]:
  assert os.path.isdir(parent_dir)
  _, _, filenames = next(os.walk(parent_dir))
  filenames.sort()
  return filenames


def get_filepaths(parent_dir: str) -> List[str]:
  names = get_filenames(parent_dir)
  res = [os.path.join(parent_dir, x) for x in names]
  return res


def get_subfolder_names(parent_dir: str) -> List[str]:
  assert os.path.isdir(parent_dir)
  _, subfolder_names, _ = next(os.walk(parent_dir))
  subfolder_names.sort()
  return subfolder_names


def get_subfolders(parent_dir: str) -> List[str]:
  """return full paths"""
  names = get_subfolder_names(parent_dir)
  res = [os.path.join(parent_dir, x) for x in names]
  return res


def console_out_len(text: str):
  res = len([c for c in text if unicodedata.combining(c) == 0])
  return res


class GenericList(list, Generic[T]):
  def save(self, file_path: str):
    data = [astuple(xi) for xi in self.items()]
    dataframe = pd.DataFrame(data)
    save_df(dataframe, file_path)

  @classmethod
  def load(cls, member_class: Type[T], file_path: str):
    data = load_df(file_path)
    data_loaded: List[T] = [member_class(*xi) for xi in data.values]
    return cls(data_loaded)

  def items(self, with_tqdm: bool = False) -> List[T]:
    if with_tqdm:
      return tqdm(self)
    else:
      return self

  def get_random_entry(self) -> T:
    idx = random.choice(range(len(self)))
    return self[idx]


def load_df(path: str) -> pd.DataFrame:
  data = pd.read_csv(path, header=None, sep=CSV_SEPERATOR)
  return data


def save_df(dataframe: pd.DataFrame, path: str):
  dataframe.to_csv(path, header=None, index=None, sep=CSV_SEPERATOR)


def get_sorted_list_from_set(unsorted_set: Set[T]) -> List[T]:
  res: List[T] = list(sorted(list(unsorted_set)))
  return res


def remove_duplicates_list_orderpreserving(l: List[str]) -> List[str]:
  result = []
  for x in l:
    if x not in result:
      result.append(x)
  assert len(result) == len(set(result))
  return result


def get_counter(l: List[List[T]]) -> Counter:
  items = []
  for sublist in l:
    items.extend(sublist)
  symbol_counter = Counter(items)
  return symbol_counter


def get_unique_items(of_list: List[Union[List[T], Set[T]]]) -> Set[T]:
  items: Set[T] = set()
  for sub_entries in of_list:
    items = items.union(set(sub_entries))
  return items


def cosine_dist_mels(a: np.ndarray, b: np.ndarray) -> float:
  a, b = make_same_dim(a, b)
  scores = []
  for channel_nr in range(a.shape[0]):
    channel_a = a[channel_nr]
    channel_b = b[channel_nr]
    score = cosine(channel_a, channel_b)
    if np.isnan(score):
      score = 1
    scores.append(score)
  score = np.mean(scores)
  #scores = cdist(pred_np, orig_np, 'cosine')
  final_score = 1 - score
  return final_score


def make_same_dim(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  dim_a = a.shape[1]
  dim_b = b.shape[1]
  diff = abs(dim_a - dim_b)
  if diff > 0:
    adding_array = np.zeros(shape=(a.shape[0], diff))
    if dim_a < dim_b:
      a = np.concatenate((a, adding_array), axis=1)
    else:
      b = np.concatenate((b, adding_array), axis=1)
  assert a.shape == b.shape
  return a, b


def get_basename(filepath: str) -> str:
  '''test.wav -> test'''
  basename, _ = os.path.splitext(os.path.basename(filepath))
  return basename


def get_parent_dirname(filepath: str):
  last_dir_name = Path(filepath).parts[-1]
  return last_dir_name


def get_chunk_name(i, chunksize, maximum):
  assert i >= 0
  assert chunksize > 0
  assert maximum >= 0
  start = i // chunksize
  start *= chunksize
  end = start + chunksize - 1
  if end > maximum:
    end = maximum
  res = f"{start}-{end}"
  return res


def stack_images_vertically(list_im, out_path):
  old_level = logging.getLogger().level
  logging.getLogger().setLevel(logging.INFO)
  images = [Image.open(i) for i in list_im]
  widths, heights = zip(*(i.size for i in images))

  total_height = sum(heights)
  max_width = max(widths)

  new_im = Image.new(
    mode='RGB',
    size=(max_width, total_height),
    color=(255, 255, 255)  # white
  )

  y_offset = 0
  for im in images:
    new_im.paste(im, (0, y_offset))
    y_offset += im.size[1]
  new_im.save(out_path)
  logging.getLogger().setLevel(old_level)


def stack_images_horizontally(list_im: List[str], out_path: str):
  old_level = logging.getLogger().level
  logging.getLogger().setLevel(logging.INFO)
  images = [Image.open(i) for i in list_im]
  widths, heights = zip(*(i.size for i in images))

  total_width = sum(widths)
  max_height = max(heights)

  new_im = Image.new(
    mode='RGB',
    size=(total_width, max_height),
    color=(255, 255, 255)  # white
  )

  x_offset = 0
  for im in images:
    new_im.paste(im, (x_offset, 0))
    x_offset += im.size[0]
  new_im.save(out_path)
  logging.getLogger().setLevel(old_level)


def create_parent_folder(file: str) -> str:
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


def read_lines(path: str) -> List[str]:
  assert os.path.isfile(path)
  with open(path, "r", encoding='utf-8') as f:
    lines = f.readlines()
  res = [x.strip("\n") for x in lines]
  return res


def read_text(path: str) -> str:
  res = '\n'.join(read_lines(path))
  return res


if __name__ == "__main__":
  pass
  #x = "/datasets/models/taco2pt_v2/debug/filelist/filelist.csv"
  #data = pd.read_csv(x, header=None, sep=_csv_separator)
  # get_speaker_count_csv(data)
