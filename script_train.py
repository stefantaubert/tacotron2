import argparse
import json
import os
from shutil import copyfile

from paths import (ds_preprocessed_file_name, ds_preprocessed_symbols_log_name,
                   ds_preprocessed_symbols_name, filelist_file_name,
                   filelist_symbols_file_name, filelist_symbols_log_file_name,
                   get_ds_dir, get_filelist_dir, get_inference_dir,
                   inference_config_file, log_inference_config, log_input_file,
                   log_map_file, log_train_config, train_config_file, log_train_map)
from script_prepare_ds import prepare
from script_split_ds import split_ds
from script_txt_pre import process_input_text
from synthesize import infer
from train import start_train, get_last_checkpoint
from train_log import reset_log
from plot_embeddings import analyse

def start_training(base_dir: str, training_dir_path: str):

  config_path = os.path.join(training_dir_path, train_config_file)

  with open(config_path, 'r', encoding='utf-8') as f:
    config = json.load(f)

  if not config["continue_training"]:
    prepare(base_dir, training_dir_path, config)
    split_ds(base_dir, training_dir_path, config)
    
  start_train(training_dir_path, config)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--debug', type=str, default="true")
  parser.add_argument('--base_dir', type=str, help='base directory')
  parser.add_argument('--training_dir', type=str)
  parser.add_argument('--continue_training', type=str)
  parser.add_argument('--seed', type=str, default=1234)
  parser.add_argument('--warm_start', type=str)
  parser.add_argument('--pretrained_path', type=str)
  parser.add_argument('--ds_name', type=str)
  parser.add_argument('--speaker', type=str)
  parser.add_argument('--train_size', type=str, default=0.9)
  parser.add_argument('--validation_size', type=str, default=1.0)
  parser.add_argument('--hparams', type=str)
  parser.add_argument('--merge_mode', type=str)
  parser.add_argument('--pretrained_model', type=str)
  parser.add_argument('--pretrained_model_symbols', type=str)
  parser.add_argument('--weight_map_mode', type=str)
  parser.add_argument('--map', type=str)

  args = parser.parse_args()

  debug = str.lower(args.debug) == 'true'

  if debug:
    args.base_dir = '/datasets/models/taco2pt_v2'
    args.training_dir = 'debug'
    train = True
    #train = False

  training_dir_path = os.path.join(args.base_dir, args.training_dir)

  print("Given arguments:")
  for arg, value in sorted(vars(args).items()):
    print("Argument {}: {}".format(arg, value))

  with open(args.config, 'r', encoding='utf-8') as f:
    config = json.load(f)

  reset_log(training_dir_path)
  log_train_config(training_dir_path, args.config)
  if config["weight_map_mode"] != 'none':
    log_train_map(training_dir_path, config["map"])
  start_training(args.base_dir, training_dir_path)
  analyse(training_dir_path)
 