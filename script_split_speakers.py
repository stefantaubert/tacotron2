import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from paths import preprocessed_file_name, test_file_name, training_file_name, validation_file_name, filelist_dir
from utils import csv_separator

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('--base_dir', type=str, help='base directory', default='/datasets/models/taco2pt_ms_learning')
  parser.add_argument('--seed', type=str, help='random seed', default='1234')
  
  args = parser.parse_args()
  seed = int(args.seed)

  filelist_dir_path = os.path.join(args.base_dir, filelist_dir)
  prepr_path = os.path.join(filelist_dir_path, preprocessed_file_name)

  data = pd.read_csv(prepr_path, header=None, sep=csv_separator)
  print(data)

  train, test = train_test_split(data, test_size=0.02, random_state=seed)
  test, val = train_test_split(test, test_size=0.5, random_state=seed)

  train.to_csv(os.path.join(filelist_dir_path, training_file_name), header=None, index=None, sep=csv_separator)
  test.to_csv(os.path.join(filelist_dir_path, test_file_name), header=None, index=None, sep=csv_separator)
  val.to_csv(os.path.join(filelist_dir_path, validation_file_name), header=None, index=None, sep=csv_separator)
  print("Dataset is splitted in train-, val- and test-set.")

  print("Total:", len(data))
  print("Trainsize:", len(train))
  print("Validationsize:", len(val))
  print("Testsize:", len(test))
