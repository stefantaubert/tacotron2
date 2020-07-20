import argparse
import os
from shutil import copyfile

import pandas as pd
from sklearn.model_selection import train_test_split

from src.script_paths import (ds_preprocessed_file_name, ds_preprocessed_symbols_name,
                   filelist_symbols_file_name, filelist_test_file_name,
                   filelist_training_file_name, filelist_validation_file_name,
                   get_ds_dir, get_filelist_dir)
from src.common.train_log import log
from src.common.utils import csv_separator, get_total_duration_min

def __save(train, training_dir_path, fn, duration_col):
  #df = train.iloc[:, __wav_symids_cols]
  total_dur_min = get_total_duration_min(train, duration_col)
  train.to_csv(os.path.join(get_filelist_dir(training_dir_path), fn), header=None, index=None, sep=csv_separator)
  log(training_dir_path, "{} => Size: {}, Duration: {:.2f}min / {:.2f}h".format(fn, len(train), total_dur_min, total_dur_min / 60))
  return total_dur_min

def split_ds(base_dir, training_dir_path: str, train_size: float, validation_size: float, seed: int, duration_col: int):
  preprocessed_path = os.path.join(get_filelist_dir(training_dir_path), ds_preprocessed_file_name)
  log(training_dir_path, "Split data into different sets from: " + preprocessed_path)
  data = pd.read_csv(preprocessed_path, header=None, sep=csv_separator)
  print(data.head())
  total_duration = 0

  train, val = train_test_split(data, train_size=train_size, random_state=seed)
  
  d = __save(train, training_dir_path, filelist_training_file_name, duration_col)
  total_duration = total_duration + d

  if validation_size < 1.0:
    test, val = train_test_split(val, test_size=validation_size, random_state=seed)
    d = __save(test, training_dir_path, filelist_test_file_name, duration_col)
    total_duration = total_duration + d
  else:
    log(training_dir_path, "Created no testset")

  d = __save(val, training_dir_path, filelist_validation_file_name, duration_col)
  total_duration = total_duration + d

  log(training_dir_path, "Total => Size: {}, Duration: {:.2f}min / {:.2f}h".format(len(data), total_duration, total_duration / 60))
  log(training_dir_path, "Dataset is splitted now.")
