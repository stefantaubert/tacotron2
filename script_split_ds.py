import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from paths import preprocessed_file, test_file, training_file, validation_file
from script_ds_pre import csv_separator

if __name__ == "__main__":
 
  parser = argparse.ArgumentParser()
  parser.add_argument('-b', '--base_dir', type=str, help='base directory', default='/datasets/models/taco2pt')
  
  args = parser.parse_args()
  prepr_path = os.path.join(args.base_dir, preprocessed_file)

  data = pd.read_csv(prepr_path, header=None, sep=csv_separator)
  print(data)

  train, test = train_test_split(data, test_size=500, random_state=1234)
  train, val = train_test_split(train, test_size=100, random_state=1234)

  #print(len(train))
  #print(len(test))
  #print(len(val))

  train.to_csv(os.path.join(args.base_dir, training_file), header=None, index=None, sep=csv_separator)
  test.to_csv(os.path.join(args.base_dir, test_file), header=None, index=None, sep=csv_separator)
  val.to_csv(os.path.join(args.base_dir, validation_file), header=None, index=None, sep=csv_separator)
  print("Dataset is splitted in train-, val- and test-set.")
