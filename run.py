import argparse
import os
import json
from shutil import copyfile
from paths import get_ds_dir, ds_preprocessed_symbols_name
from script_split_ds import split_ds
from train import start_train

def start_training(base_dir: str, training_dir_path: str, config: dict):
  split_ds(base_dir, training_dir_path, config)
  start_train(base_dir, training_dir_path, config)

def start_inference(base_dir: str, training_dir: str, config: dict):
  pass

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--base_dir', type=str, help='base directory')
  parser.add_argument('--training_dir', type=str)
  parser.add_argument('--mode', type=str, help="train,infer")
  parser.add_argument('--config', type=str)
  parser.add_argument('--debug', type=str, default="true")

  args = parser.parse_args()

  debug = str.lower(args.debug) == 'true'

  if debug:
    args.base_dir = '/datasets/models/taco2pt_v2'
    args.training_dir = 'debug'
    args.mode = 'train'
    args.config = "configs/ljs_en/train.json"
  
  print("Given arguments:")
  for arg, value in sorted(vars(args).items()):
    print("Argument {}: {}".format(arg, value))

  with open(args.config, 'r', encoding='utf-8') as f:
    config = json.load(f)

  training_dir_path = os.path.join(args.base_dir, args.training_dir)

  if args.mode == 'train':
    start_training(args.base_dir, training_dir_path, config)
  else:
    start_inference(args.base_dir, training_dir_path, config)
