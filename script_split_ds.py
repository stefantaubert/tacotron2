import argparse
import os
from shutil import copyfile

import pandas as pd
from sklearn.model_selection import train_test_split

from paths import ds_preprocessed_symbols_name, ds_preprocessed_file_name, ds_preprocessed_symbols_log_name, filelist_symbols_log_file_name, filelist_symbols_file_name, filelist_test_file_name, filelist_training_file_name, filelist_validation_file_name, get_filelist_dir, get_ds_dir
from utils import csv_separator

def split_ds(base_dir, training_dir_path: str, config: dict):
  speaker_dir_path = get_ds_dir(base_dir, config["ds_name"], config["speaker"])

  preprocessed_path = os.path.join(speaker_dir_path, ds_preprocessed_file_name)

  data = pd.read_csv(preprocessed_path, header=None, sep=csv_separator)
  print(data)

  # train, test = train_test_split(data, test_size=500, random_state=1234)
  # train, val = train_test_split(train, test_size=100, random_state=1234)

  # train, test = train_test_split(data, test_size=0.04, random_state=1234)
  # train, val = train_test_split(train, test_size=0.01, random_state=1234)

  train, test = train_test_split(data, test_size=0.02, random_state=config["seed"])
  test, val = train_test_split(test, test_size=0.5, random_state=config["seed"])

  train.to_csv(os.path.join(get_filelist_dir(training_dir_path), filelist_training_file_name), header=None, index=None, sep=csv_separator)
  test.to_csv(os.path.join(get_filelist_dir(training_dir_path), filelist_test_file_name), header=None, index=None, sep=csv_separator)
  val.to_csv(os.path.join(get_filelist_dir(training_dir_path), filelist_validation_file_name), header=None, index=None, sep=csv_separator)
  print("Dataset is splitted in train-, val- and test-set.")

  print("Total:", len(data))
  print("Trainsize:", len(train))
  print("Validationsize:", len(val))
  print("Testsize:", len(test))
