import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from paths import preprocessed_file, test_file, training_file, validation_file
from script_ds_pre import csv_separator

if __name__ == "__main__":
 
  parser = argparse.ArgumentParser()
  parser.add_argument('--base_dir', type=str, help='base directory', default='/datasets/models/taco2pt_testing')
  parser.add_argument('--seed', type=str, help='random seed', default='1234')
  
  args = parser.parse_args()
  seed = int(args.seed)
  prepr_path = os.path.join(args.base_dir, preprocessed_file)

  data = pd.read_csv(prepr_path, header=None, sep=csv_separator)
  print(data)

  # train, test = train_test_split(data, test_size=500, random_state=1234)
  # train, val = train_test_split(train, test_size=100, random_state=1234)

  # train, test = train_test_split(data, test_size=0.04, random_state=1234)
  # train, val = train_test_split(train, test_size=0.01, random_state=1234)

  train, test = train_test_split(data, test_size=0.02, random_state=seed)
  test, val = train_test_split(test, test_size=0.5, random_state=seed)

  print("Total:", len(data))
  print("Trainsize:", len(train))
  print("Validationsize:", len(val))
  print("Testsize:", len(test))

  train.to_csv(os.path.join(args.base_dir, training_file), header=None, index=None, sep=csv_separator)
  test.to_csv(os.path.join(args.base_dir, test_file), header=None, index=None, sep=csv_separator)
  val.to_csv(os.path.join(args.base_dir, validation_file), header=None, index=None, sep=csv_separator)
  print("Dataset is splitted in train-, val- and test-set.")
