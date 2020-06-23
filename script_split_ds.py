import argparse
import os
from shutil import copyfile

import pandas as pd
from sklearn.model_selection import train_test_split

from paths import ds_preprocessed_symbols_name, ds_preprocessed_file_name, filelist_symbols_log_file_name, filelist_symbols_file_name, filelist_test_file_name, filelist_training_file_name, filelist_validation_file_name, get_filelist_dir, get_ds_dir
from utils import csv_separator
from train_log import log

__duration_col = [3]

def __save(train, training_dir_path, fn):
  #df = train.iloc[:, __wav_symids_cols]
  total_dur_min = float(train.iloc[:, __duration_col].sum(axis=0)) / 60
  train.to_csv(os.path.join(get_filelist_dir(training_dir_path), fn), header=None, index=None, sep=csv_separator)
  log(training_dir_path, "{} => Size: {}, Duration: {:.2f}min".format(fn, len(train), total_dur_min))
  return total_dur_min

def split_ds(base_dir, training_dir_path: str, config: dict):
  preprocessed_path = os.path.join(get_filelist_dir(training_dir_path), ds_preprocessed_file_name)
  log(training_dir_path, "Split data into different sets from: " + preprocessed_path)
  data = pd.read_csv(preprocessed_path, header=None, sep=csv_separator)
  print(data.head())
  total_duration = 0

  train, val = train_test_split(data, train_size=config["train_size"], random_state=config["seed"])
  
  d = __save(train, training_dir_path, filelist_training_file_name)
  total_duration = total_duration + d

  # all_cols = list(range(len(train.columns)))
  # for i in wav_symids_cols:
  #   all_cols.remove(i)
  #df2 = train.iloc[:, all_cols]

  #df = train.iloc[:, __wav_symids_cols]
  #df2 = train.iloc[:, all_cols]
  #total_dur_min = train.iloc[:, __duration_col].sum() / 60
  #df.to_csv(os.path.join(get_filelist_dir(training_dir_path), filelist_training_file_name), header=None, index=None, sep=csv_separator)
  #log(training_dir_path, "Trainsize: {}, Duration: {:.2f}".format(len(train), total_dur_min))

  if config["validation_size"] < 1.0:
    test, val = train_test_split(val, test_size=config["validation_size"], random_state=config["seed"])
    d = __save(test, training_dir_path, filelist_test_file_name)
    total_duration = total_duration + d
  else:
    log(training_dir_path, "Created no testset")

  d = __save(val, training_dir_path, filelist_validation_file_name)
  total_duration = total_duration + d

  log(training_dir_path, "Total => Size: {}, Duration: {:.2f}min".format(len(data), total_duration))
  log(training_dir_path, "Dataset is splitted now.")

if __name__ == "__main__":
  split_ds