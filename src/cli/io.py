import datetime
import os
from shutil import copyfile
from typing import List

import matplotlib.pylab as plt
import numpy as np
from tqdm import tqdm

from src.core.common import (Language, float_to_wav, get_basename,
                             get_custom_or_last_checkpoint, get_parent_dirname,
                             get_subdir, parse_json, plot_melspec,
                             stack_images_vertically)
from src.core.pre import PreparedDataList, SpeakersIdDict, SymbolConverter

def get_train_root_dir(base_dir: str, train_name: str, create: bool):
  return get_subdir(base_dir, os.path.join("train", train_name), create)

#region Training

def _get_train_csv(base_dir: str, train_name: str):
  return os.path.join(get_train_root_dir(base_dir, train_name, create=True), "train.csv")

def _get_test_csv(base_dir: str, train_name: str):
  return os.path.join(get_train_root_dir(base_dir, train_name, create=True), "test.csv")

def _get_val_csv(base_dir: str, train_name: str):
  return os.path.join(get_train_root_dir(base_dir, train_name, create=True), "validation.csv")

def get_train_log_dir(base_dir: str, train_name: str):
  train_dir = get_train_root_dir(base_dir, train_name, create=True)
  return get_subdir(train_dir, "logs", create=True)

def get_train_log_file(base_dir: str, train_name: str):
  return os.path.join(get_train_log_dir(base_dir, train_name), "log.txt")

def get_checkpoints_dir(base_dir: str, train_name: str):
  train_dir = get_train_root_dir(base_dir, train_name, create=True)
  return get_subdir(train_dir, "checkpoints", create=True)

def save_trainset(base_dir: str, train_name: str, dataset: PreparedDataList):
  path = _get_train_csv(base_dir, train_name)
  dataset.save(path)

def load_trainset(base_dir: str, train_name: str) -> PreparedDataList:
  path = _get_train_csv(base_dir, train_name)
  return PreparedDataList.load(path)

def save_testset(base_dir: str, train_name: str, dataset: PreparedDataList):
  path = _get_test_csv(base_dir, train_name)
  dataset.save(path)

def load_testset(base_dir: str, train_name: str) -> PreparedDataList:
  path = _get_test_csv(base_dir, train_name)
  return PreparedDataList.load(path)
  
def save_valset(base_dir: str, train_name: str, dataset: PreparedDataList):
  path = _get_val_csv(base_dir, train_name)
  dataset.save(path)

def load_valset(base_dir: str, train_name: str) -> PreparedDataList:
  path = _get_val_csv(base_dir, train_name)
  return PreparedDataList.load(path)
  
#endregion

#region Inference

def get_inference_root_dir(base_dir: str, train_name: str):
  train_dir = get_train_root_dir(base_dir, train_name, create=True)
  return get_subdir(train_dir, "inference", create=True)

def get_infer_log(infer_dir: str):
  return os.path.join(infer_dir, f"{get_parent_dirname(infer_dir)}.txt")

def save_infer_wav(infer_dir: str, sampling_rate: int, wav: np.ndarray):
  path = os.path.join(infer_dir, f"{get_parent_dirname(infer_dir)}.wav")
  float_to_wav(wav, path, normalize=True, sample_rate=sampling_rate)

def save_infer_plot(infer_dir: str, mel: np.ndarray):
  plot_melspec(mel, title=get_parent_dirname(infer_dir))
  path = os.path.join(infer_dir, f"{get_parent_dirname(infer_dir)}.png")
  plt.savefig(path, bbox_inches='tight')
  return path
  
#endregion

#region Validation

def _get_validation_root_dir(base_dir: str, train_name: str):
  train_dir = get_train_root_dir(base_dir, train_name, create=True)
  return get_subdir(train_dir, "validation", create=True)

def get_val_dir(base_dir: str, train_name: str, entry_id: int, iteration: int):
  subdir_name = "{}_id-{}_it-{}".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), entry_id, iteration)
  return get_subdir(_get_validation_root_dir(base_dir, train_name), subdir_name, create=True)

def save_val_plot(val_dir: str, mel):
  parent_dir = get_parent_dirname(val_dir)
  plot_melspec(mel, title=parent_dir)
  path = os.path.join(val_dir, f"{parent_dir}.png")
  plt.savefig(path, bbox_inches='tight')

def save_val_orig_plot(val_dir: str, mel):
  parent_dir = get_parent_dirname(val_dir)
  plot_melspec(mel, title=parent_dir)
  path = os.path.join(val_dir, f"{parent_dir}_orig.png")
  plt.savefig(path, bbox_inches='tight')

def save_val_comparison(val_dir: str):
  parent_dir = get_parent_dirname(val_dir)
  path1 = os.path.join(val_dir, f"{parent_dir}.png")
  path2 = os.path.join(val_dir, f"{parent_dir}_orig.png")
  assert os.path.exists(path1)
  assert os.path.exists(path2)
  path = os.path.join(val_dir, f"{parent_dir}_comp.png")
  stack_images_vertically([path1, path2], path)

def save_val_wav(val_dir: str, sampling_rate: int, wav) -> str:
  path = os.path.join(val_dir, f"{get_parent_dirname(val_dir)}.wav")
  float_to_wav(wav, path, normalize=True, sample_rate=sampling_rate)
  return path

def save_val_orig_wav(val_dir: str, wav_path: str):
  path = os.path.join(val_dir, f"{get_parent_dirname(val_dir)}_orig.wav")
  copyfile(wav_path, path)

def get_val_log(val_dir: str):
  return os.path.join(val_dir, f"{get_parent_dirname(val_dir)}.txt")

#endregion
