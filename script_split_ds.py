import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from paths import preprocessed_file_name, pre_ds_thchs_dir, pre_ds_ljs_dir, test_file_name, training_file_name, validation_file_name
from utils import csv_separator

if __name__ == "__main__":
 
  parser = argparse.ArgumentParser()
  parser.add_argument('--base_dir', type=str, help='base directory', default='/datasets/models/taco2pt_ms')
  parser.add_argument('--seed', type=str, help='random seed', default='1234')
  parser.add_argument('--ds', type=str, required=False, default='ljs', help='thchs or ljs')
  parser.add_argument('--speaker', type=str, required=False, default='1', help='speaker')
  
  args = parser.parse_args()
  seed = int(args.seed)

  if args.ds == 'ljs':
    ds = pre_ds_ljs_dir
  elif args.ds == 'thchs':
    ds = pre_ds_thchs_dir
  else: 
    raise Exception()

  speaker_dir = os.path.join(args.base_dir, ds, args.speaker)
  prepr_path = os.path.join(speaker_dir, preprocessed_file_name)

  data = pd.read_csv(prepr_path, header=None, sep=csv_separator)
  print(data)

  # train, test = train_test_split(data, test_size=500, random_state=1234)
  # train, val = train_test_split(train, test_size=100, random_state=1234)

  # train, test = train_test_split(data, test_size=0.04, random_state=1234)
  # train, val = train_test_split(train, test_size=0.01, random_state=1234)

  train, test = train_test_split(data, test_size=0.02, random_state=seed)
  test, val = train_test_split(test, test_size=0.5, random_state=seed)

  train.to_csv(os.path.join(speaker_dir, training_file_name), header=None, index=None, sep=csv_separator)
  test.to_csv(os.path.join(speaker_dir, test_file_name), header=None, index=None, sep=csv_separator)
  val.to_csv(os.path.join(speaker_dir, validation_file_name), header=None, index=None, sep=csv_separator)
  print("Dataset is splitted in train-, val- and test-set.")

  print("Total:", len(data))
  print("Trainsize:", len(train))
  print("Validationsize:", len(val))
  print("Testsize:", len(test))
