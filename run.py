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
from script_merge_speakers import merge_speakers
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

  speaker_dir_path = get_ds_dir(base_dir, config["ds_name"], config["speaker"])
  if not config["continue_training"]:
    if config["merge_symbols"]:
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
    
  start_train(training_dir_path, config)

def start_inference(training_dir_path: str, infer_dir_path: str):
  config_path = os.path.join(infer_dir_path, inference_config_file)
  
  with open(config_path, 'r', encoding='utf-8') as f:
    config = json.load(f)

  process_input_text(training_dir_path, infer_dir_path, config)
  infer(training_dir_path, infer_dir_path, config)

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
    if True:
      args.mode = 'train'
      args.config = "configs/thchs_toneless_ipa/train.json"
    else:
      args.mode = 'infer'
      args.config = "configs/thchs_toneless_ipa/infer.json"

  training_dir_path = os.path.join(args.base_dir, args.training_dir)

  print("Given arguments:")
  for arg, value in sorted(vars(args).items()):
    print("Argument {}: {}".format(arg, value))

  with open(args.config, 'r', encoding='utf-8') as f:
    config = json.load(f)

  if args.mode == 'train':
    reset_log(training_dir_path)
    log_train_config(training_dir_path, args.config)
    if config["merge_symbols"] and config["map_pretrained_weights"]:
      log_train_map(training_dir_path, config["map"])
    start_training(args.base_dir, training_dir_path)
    analyse(training_dir_path)
  else:
    input_file = config["text"]
    print("Infering text from:", input_file)
    input_name = os.path.splitext(os.path.basename(input_file))[0]
    checkpoint = get_last_checkpoint(training_dir_path)
    if config["custom_checkpoint"] != '':
      checkpoint = config["custom_checkpoint"]
    infer_dir_path = get_inference_dir(training_dir_path, checkpoint, input_name)
    log_inference_config(infer_dir_path, args.config)
    log_input_file(infer_dir_path, input_file)

    if config["map"] != '':
      print("Using mapping from:", config["map"])
      log_map_file(infer_dir_path, config["map"])
    else:
      print("Using no mapping.")

    start_inference(training_dir_path, infer_dir_path)
