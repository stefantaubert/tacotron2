import argparse
import os
from shutil import copyfile

import pandas as pd
from sklearn.model_selection import train_test_split

from paths import ds_preprocessed_symbols_name, ds_preprocessed_file_name, ds_preprocessed_symbols_log_name, filelist_symbols_log_file_name, filelist_symbols_file_name, filelist_test_file_name, filelist_training_file_name, filelist_validation_file_name, get_filelist_dir, get_ds_dir
from utils import csv_separator
from train_log import log

def split_ds(base_dir, training_dir_path: str, config: dict):
  preprocessed_path = os.path.join(get_filelist_dir(training_dir_path), ds_preprocessed_file_name)
  log(training_dir_path, "Split data into different sets from: " + preprocessed_path)
  data = pd.read_csv(preprocessed_path, header=None, sep=csv_separator)
  print(data.head())

  train, val = train_test_split(data, train_size=config["train_size"], random_state=config["seed"])
  train.to_csv(os.path.join(get_filelist_dir(training_dir_path), filelist_training_file_name), header=None, index=None, sep=csv_separator)
  log(training_dir_path, "Trainsize:" + str(len(train)))

  if config["validation_size"] < 1.0:
    test, val = train_test_split(val, test_size=config["validation_size"], random_state=config["seed"])
    test.to_csv(os.path.join(get_filelist_dir(training_dir_path), filelist_test_file_name), header=None, index=None, sep=csv_separator)
    log(training_dir_path, "Testsize:" + str(len(test)))
  else:
    log(training_dir_path, "Created no testset")

  val.to_csv(os.path.join(get_filelist_dir(training_dir_path), filelist_validation_file_name), header=None, index=None, sep=csv_separator)
  log(training_dir_path, "Validationsize:" + str(len(val)))
  log(training_dir_path, "Total:" + str(len(data)))
  log(training_dir_path, "Dataset is splitted now.")
