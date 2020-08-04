from src.paths import (ds_preprocessed_file_name, ds_preprocessed_symbols_name,
                       filelist_file_log_name, filelist_file_name,
                       filelist_speakers_name, filelist_symbols_file_name,
                       filelist_test_file_name, filelist_training_file_name,
                       filelist_validation_file_name,
                       filelist_weights_file_name, get_all_speakers_path,
                       get_ds_dir, get_filelist_dir, train_map_file)
import os
from src.common.utils import load_csv, save_csv, save_json
from collections import OrderedDict
from src.common.train_log import log
from src.common.utils import get_total_duration_min, parse_json
import numpy as np

__basename_idx = 0
__wavepath_idx = 1
__duration_idx = 2

def get_basename(values):
  return values[__basename_idx]

def get_wavepath(values):
  return values[__wavepath_idx]

def get_duration(values):
  return values[__duration_idx]

def to_values(basename, wav_path, duration):
  return (basename, wav_path, duration)

def save_trainset(training_dir_path, dataset: list):
  dest_path = os.path.join(get_filelist_dir(training_dir_path), filelist_training_file_name)
  df = save_csv(dataset, dest_path)
  total_dur_min = get_total_duration_min(df, __duration_idx)
  log(training_dir_path, "{} => Size: {}, Duration: {:.2f}min / {:.2f}h".format(filelist_training_file_name, len(df), total_dur_min, total_dur_min / 60))

def save_testset(training_dir_path, dataset: list):
  dest_path = os.path.join(get_filelist_dir(training_dir_path), filelist_test_file_name)
  df = save_csv(dataset, dest_path)
  total_dur_min = get_total_duration_min(df, __duration_idx)
  log(training_dir_path, "{} => Size: {}, Duration: {:.2f}min / {:.2f}h".format(filelist_test_file_name, len(df), total_dur_min, total_dur_min / 60))

def save_validationset(training_dir_path, dataset: list):
  dest_path = os.path.join(get_filelist_dir(training_dir_path), filelist_validation_file_name)
  df = save_csv(dataset, dest_path)
  total_dur_min = get_total_duration_min(df, __duration_idx)
  log(training_dir_path, "{} => Size: {}, Duration: {:.2f}min / {:.2f}h".format(filelist_validation_file_name, len(df), total_dur_min, total_dur_min / 60))

def save_wholeset(training_dir_path, dataset: list):
  dest_path = os.path.join(get_filelist_dir(training_dir_path), filelist_file_name)
  df = save_csv(dataset, dest_path)
  total_dur_min = get_total_duration_min(df, __duration_idx)
  log(training_dir_path, "{} => Size: {}, Duration: {:.2f}min / {:.2f}h".format(filelist_file_name, len(df), total_dur_min, total_dur_min / 60))

def parse_traindata(training_dir_path: str):
  dest_path = os.path.join(get_filelist_dir(training_dir_path), filelist_training_file_name)
  return load_csv(dest_path).values

def parse_validationset(training_dir_path: str):
  dest_path = os.path.join(get_filelist_dir(training_dir_path), filelist_validation_file_name)
  return load_csv(dest_path).values
