import argparse
import os
import json
from shutil import copyfile
from paths import get_ds_dir, ds_preprocessed_symbols_name, get_filelist_dir, filelist_symbols_file_name, filelist_symbols_log_file_name, ds_preprocessed_symbols_log_name, ds_preprocessed_file_name, filelist_file_name, log_train_config, log_inference_config
from script_split_ds import split_ds
from train import start_train
from script_merge_speakers import merge_speakers

def start_training(base_dir: str, training_dir_path: str, config: dict):
  speaker_dir_path = get_ds_dir(base_dir, config["ds_name"], config["speaker"])

  if config["use_pretrained_weights"]:
    merge_speakers(base_dir, training_dir_path, config)
  else:
    # copy symbols.json
    a = os.path.join(speaker_dir_path, ds_preprocessed_symbols_name)
    b = os.path.join(get_filelist_dir(training_dir_path), filelist_symbols_file_name)
    copyfile(a, b)

    # copy symbols.log
    a = os.path.join(speaker_dir_path, ds_preprocessed_symbols_log_name)
    b = os.path.join(get_filelist_dir(training_dir_path), filelist_symbols_log_file_name)
    copyfile(a, b)

    # copy filelist.csv
    a = os.path.join(speaker_dir_path, ds_preprocessed_file_name)
    b = os.path.join(get_filelist_dir(training_dir_path), filelist_file_name)
    copyfile(a, b)

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
    args.config = "configs/thchs_toneless_ipa/train.json"
  
  print("Given arguments:")
  for arg, value in sorted(vars(args).items()):
    print("Argument {}: {}".format(arg, value))

  with open(args.config, 'r', encoding='utf-8') as f:
    config = json.load(f)

  training_dir_path = os.path.join(args.base_dir, args.training_dir)

  if args.mode == 'train':
    log_train_config(training_dir_path, args.config)
    start_training(args.base_dir, training_dir_path, config)
  else:
    #log_inference_config(training_dir_path, config)
    start_inference(args.base_dir, training_dir_path, config)
