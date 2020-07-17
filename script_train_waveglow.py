import argparse
import json
import os
from shutil import copyfile

from waveglow.hparams import create_hparams
from script_paths import (ds_preprocessed_file_name,
                   ds_preprocessed_symbols_name, filelist_file_name,
                   filelist_symbols_file_name,
                   filelist_weights_file_name, get_ds_dir, get_filelist_dir,
                   inference_config_file,
                   log_inference_config, log_input_file, log_map_file,
                   log_train_config, log_train_map, train_config_file,
                   train_map_file)
from script_plot_embeddings import analyse
from tacotron.prepare_ds import prepare
from tacotron.prepare_ds_ms import prepare as prepare_ms
from common.split_ds import split_ds
from tacotron.txt_pre import process_input_text
from tacotron.synthesize import infer
from waveglow.train import get_last_checkpoint, start_train
from common.train_log import reset_log
from common.utils import args_to_str

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--no_debugging', action='store_true')
  parser.add_argument('--base_dir', type=str, help='base directory')
  parser.add_argument('--training_dir', type=str)
  parser.add_argument('--continue_training', action='store_true')
  parser.add_argument('--seed', type=str, default=1234)
  parser.add_argument('--pretrained_path', type=str)
  parser.add_argument('--speakers', type=str, help="ds_name,speaker_id;... or ds_name,all;...")
  parser.add_argument('--train_size', type=float, default=0.9)
  parser.add_argument('--validation_size', type=float, default=1.0)
  parser.add_argument('--hparams', type=str)

  args = parser.parse_args()

  if not args.no_debugging:
    args.base_dir = '/datasets/models/taco2pt_v2'
    args.training_dir = 'wg_debug'
    args.speakers = 'ljs_en_v2,all'
    args.hparams = 'batch_size=4,iters_per_checkpoint=3'
    #args.continue_training = True

  if not args.base_dir:
    raise Exception("Argument 'base_dir' is required.")
  elif not args.training_dir:
    raise Exception("Argument 'training_dir' is required.")

  hparams = create_hparams(args.hparams)
  training_dir_path = os.path.join(args.base_dir, args.training_dir)

  if not args.continue_training:
    reset_log(training_dir_path)
    prepare_ms(args.base_dir, training_dir_path, speakers=args.speakers, pretrained_model=args.pretrained_model, weight_map_mode=args.weight_map_mode, hparams=hparams, pretrained_model_symbols=args.pretrained_model_symbols)
    split_ds(args.base_dir, training_dir_path, train_size=args.train_size, validation_size=args.validation_size, seed=args.seed)
    
  weights_path = os.path.join(get_filelist_dir(training_dir_path), filelist_weights_file_name)
  use_weights_map = os.path.exists(weights_path)
  start_train(training_dir_path, hparams=hparams, use_weights=use_weights_map, pretrained_path=args.pretrained_path, warm_start=args.warm_start, continue_training=args.continue_training)

  analyse(training_dir_path)
