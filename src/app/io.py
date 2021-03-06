import datetime
import os
from shutil import copyfile
from typing import Optional, Tuple

import matplotlib.pylab as plt
import numpy as np

from src.app.pre.prepare import load_filelist
from src.core.common.audio import float_to_wav
from src.core.common.mel_plot import plot_melspec
from src.core.common.utils import (get_parent_dirname, get_subdir, parse_json,
                                   save_json, stack_images_vertically)
from src.core.pre.merge_ds import (PreparedData, PreparedDataList,
                                   split_prepared_data_train_test_val)

# region Training

_train_csv = "train.csv"
_test_csv = "test.csv"
_val_csv = "validation.csv"
_settings_json = "settings.json"


def get_train_root_dir(base_dir: str, model_name: str, create: bool):
  return get_subdir(base_dir, model_name, create)


def get_train_logs_dir(train_dir: str):
  return get_subdir(train_dir, "logs", create=True)


def get_train_log_file(logs_dir: str):
  return os.path.join(logs_dir, "log.txt")


def get_train_checkpoints_log_file(logs_dir: str):
  return os.path.join(logs_dir, "log_checkpoints.txt")


def get_checkpoints_dir(train_dir: str):
  return get_subdir(train_dir, "checkpoints", create=True)


def save_trainset(train_dir: str, dataset: PreparedDataList):
  path = os.path.join(train_dir, _train_csv)
  dataset.save(path)


def load_trainset(train_dir: str) -> PreparedDataList:
  path = os.path.join(train_dir, _train_csv)
  return PreparedDataList.load(PreparedData, path)


def save_testset(train_dir: str, dataset: PreparedDataList):
  path = os.path.join(train_dir, _test_csv)
  dataset.save(path)


def load_testset(train_dir: str) -> PreparedDataList:
  path = os.path.join(train_dir, _test_csv)
  return PreparedDataList.load(PreparedData, path)


def save_valset(train_dir: str, dataset: PreparedDataList):
  path = os.path.join(train_dir, _val_csv)
  dataset.save(path)


def load_valset(train_dir: str) -> PreparedDataList:
  path = os.path.join(train_dir, _val_csv)
  return PreparedDataList.load(PreparedData, path)


def load_prep_name(train_dir: str) -> str:
  path = os.path.join(train_dir, _settings_json)
  res = parse_json(path)
  return res["prep_name"]


def save_prep_name(train_dir: str, prep_name: Optional[str]):
  settings = {
    "prep_name": prep_name
  }
  path = os.path.join(train_dir, _settings_json)
  save_json(path, settings)


def split_dataset(prep_dir: str, train_dir: str, test_size: float = 0.01, validation_size: float = 0.05, split_seed: int = 1234):
  wholeset = load_filelist(prep_dir)
  trainset, testset, valset = split_prepared_data_train_test_val(
    wholeset, test_size=test_size, validation_size=validation_size, seed=split_seed, shuffle=True)
  save_trainset(train_dir, trainset)
  save_testset(train_dir, testset)
  save_valset(train_dir, valset)
  return trainset, valset

# endregion

# region Inference


def get_inference_root_dir(train_dir: str):
  return get_subdir(train_dir, "inference", create=True)


def get_infer_log(infer_dir: str):
  return os.path.join(infer_dir, f"{get_parent_dirname(infer_dir)}.txt")


def save_infer_wav(infer_dir: str, sampling_rate: int, wav: np.ndarray):
  path = os.path.join(infer_dir, f"{get_parent_dirname(infer_dir)}.wav")
  float_to_wav(wav, path, sample_rate=sampling_rate)


def save_infer_plot(infer_dir: str, mel: np.ndarray):
  plot_melspec(mel, title=get_parent_dirname(infer_dir))
  path = os.path.join(infer_dir, f"{get_parent_dirname(infer_dir)}.png")
  plt.savefig(path, bbox_inches='tight')
  plt.close()
  return path

# endregion

# region Validation


def _get_validation_root_dir(train_dir: str):
  return get_subdir(train_dir, "validation", create=True)


def get_val_dir(train_dir: str, entry: PreparedData, iteration: int):
  subdir_name = f"{datetime.datetime.now():%Y-%m-%d,%H-%M-%S},id={entry.entry_id},speaker={entry.speaker_id},it={iteration}"
  return get_subdir(_get_validation_root_dir(train_dir), subdir_name, create=True)


def save_val_plot(val_dir: str, mel):
  parent_dir = get_parent_dirname(val_dir)
  plot_melspec(mel, title=parent_dir)
  path = os.path.join(val_dir, f"{parent_dir}.png")
  plt.savefig(path, bbox_inches='tight')
  plt.close()


def save_val_orig_plot(val_dir: str, mel):
  parent_dir = get_parent_dirname(val_dir)
  plot_melspec(mel, title=parent_dir)
  path = os.path.join(val_dir, f"{parent_dir}_orig.png")
  plt.savefig(path, bbox_inches='tight')
  plt.close()


def save_val_comparison(val_dir: str):
  parent_dir = get_parent_dirname(val_dir)
  path1 = os.path.join(val_dir, f"{parent_dir}_orig.png")
  path2 = os.path.join(val_dir, f"{parent_dir}.png")
  assert os.path.exists(path1)
  assert os.path.exists(path2)
  path = os.path.join(val_dir, f"{parent_dir}_comp.png")
  stack_images_vertically([path1, path2], path)


def save_val_wav(val_dir: str, sampling_rate: int, wav) -> str:
  path = os.path.join(val_dir, f"{get_parent_dirname(val_dir)}.wav")
  float_to_wav(wav, path, sample_rate=sampling_rate)
  return path


def save_val_orig_wav(val_dir: str, wav_path: str):
  path = os.path.join(val_dir, f"{get_parent_dirname(val_dir)}_orig.wav")
  copyfile(wav_path, path)


def get_val_log(val_dir: str):
  return os.path.join(val_dir, f"{get_parent_dirname(val_dir)}.txt")

# endregion
