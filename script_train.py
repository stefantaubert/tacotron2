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
from script_prepare_ds_ms import prepare as prepare_ms
from script_prepare_ds import prepare
from script_split_ds import split_ds
from script_txt_pre import process_input_text
from synthesize import infer
from train import start_train, get_last_checkpoint
from plot_embeddings import analyse
from utils import args_to_str
from train_log import reset_log

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--no_debugging', action='store_true')
  parser.add_argument('--base_dir', type=str, help='base directory')
  parser.add_argument('--training_dir', type=str)
  parser.add_argument('--continue_training', action='store_true')
  parser.add_argument('--seed', type=str, default=1234)
  parser.add_argument('--warm_start', action='store_true')
  parser.add_argument('--pretrained_path', type=str)
  #parser.add_argument('--ds_name', type=str)
  #parser.add_argument('--speaker', type=str)
  parser.add_argument('--speakers', type=str)
  parser.add_argument('--train_size', type=str, default=0.9)
  parser.add_argument('--validation_size', type=str, default=1.0)
  parser.add_argument('--hparams', type=str)
  parser.add_argument('--merge_mode', type=str)
  parser.add_argument('--pretrained_model', type=str)
  parser.add_argument('--pretrained_model_symbols', type=str)
  parser.add_argument('--weight_map_mode', type=str)
  parser.add_argument('--map', type=str)

  args = parser.parse_args()

  if not args.no_debugging:
    args.base_dir = '/datasets/models/taco2pt_v2'
    args.speakers = 'ljs_ipa_v2,1'
    args.hparams = 'batch_size=26,iters_per_checkpoint=500'
    args.training_dir = 'debug_ljs_ms'

  training_dir_path = os.path.join(args.base_dir, args.training_dir)

  reset_log(training_dir_path)
  log_train_config(training_dir_path, args)

  use_weights = bool(args.weight_map_mode)
  if use_weights:
    log_train_map(training_dir_path, args.map)

  if not args.continue_training:
    #prepare(args.base_dir, training_dir_path, merge_mode=args.merge_mode, pretrained_model_symbols=args.pretrained_model_symbols, ds_name=args.ds_name, speaker=args.speaker, pretrained_model=args.pretrained_model, weight_map_mode=args.weight_map_mode, hparams=args.hparams)
    prepare_ms(args.base_dir, training_dir_path, speakers=args.speakers)
    split_ds(args.base_dir, training_dir_path, train_size=args.train_size, validation_size=args.validation_size, seed=args.seed)
    
  #start_train(training_dir_path, hparams=args.hparams, use_weights=use_weights, pretrained_path=args.pretrained_path, warm_start=args.warm_start, continue_training=args.continue_training)
  start_train(training_dir_path, hparams=args.hparams, use_weights=use_weights, pretrained_path=args.pretrained_path, warm_start=args.warm_start, continue_training=args.continue_training, speakers=args.speakers)

  analyse(training_dir_path)
 