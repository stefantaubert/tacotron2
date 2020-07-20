import argparse
import json
import os
from shutil import copyfile

from src.tacotron.hparams import create_hparams
from src.script_paths import (ds_preprocessed_file_name,
                   ds_preprocessed_symbols_name, filelist_file_name,
                   filelist_symbols_file_name,
                   filelist_weights_file_name, get_ds_dir, get_filelist_dir,
                   inference_config_file,
                   log_inference_config, log_input_file, log_map_file,
                   log_train_config, log_train_map, train_config_file,
                   train_map_file)
from src.tacotron.script_plot_embeddings import analyse
from src.tacotron.prepare_ds import prepare
from src.tacotron.prepare_ds_ms import prepare as prepare_ms
from src.common.split_ds import split_ds
from src.tacotron.txt_pre import process_input_text
from src.tacotron.synthesize import infer
from src.tacotron.train import get_last_checkpoint, start_train
from src.common.train_log import reset_log
from src.common.utils import args_to_str, duration_col

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--no_debugging', action='store_true')
  parser.add_argument('--base_dir', type=str, help='base directory')
  parser.add_argument('--training_dir', type=str)
  parser.add_argument('--continue_training', action='store_true')
  parser.add_argument('--seed', type=str, default=1234)
  parser.add_argument('--warm_start', action='store_true')
  parser.add_argument('--pretrained_path', type=str)
  parser.add_argument('--speakers', type=str, help="ds_name,speaker_id;... or ds_name,all;...")
  parser.add_argument('--train_size', type=float, default=0.9)
  parser.add_argument('--validation_size', type=float, default=1.0)
  parser.add_argument('--hparams', type=str)
  parser.add_argument('--pretrained_model', type=str)
  parser.add_argument('--pretrained_model_symbols', type=str)
  parser.add_argument('--weight_map_mode', type=str, choices=['', 'same_symbols_only', 'use_map'])
  parser.add_argument('--map', type=str)

  args = parser.parse_args()

  if not args.no_debugging:
    args.base_dir = '/datasets/models/taco2pt_v2'
    args.training_dir = 'debug'
    args.speakers = 'thchs_v5,all'
    args.hparams = 'batch_size=20,iters_per_checkpoint=3,ignore_layers=[embedding.weight, speakers_embedding.weight]'
    args.pretrained_path = "/datasets/models/pretrained/ljs_ipa_scratch_80000"
    args.warm_start = True
    #args.weight_map_mode = 'same_symbols_only'
    #args.weight_map_mode = 'use_map'
    #args.map = "maps/weights/chn_en_v1.json"
    #args.pretrained_model = "/datasets/models/pretrained/ljs_ipa_scratch_80000"
    #args.pretrained_model_symbols = "/datasets/models/pretrained/ljs_ipa_scratch.json"
    #args.continue_training = True

  if not args.base_dir:
    raise Exception("Argument 'base_dir' is required.")
  elif not args.training_dir:
    raise Exception("Argument 'training_dir' is required.")

  hparams = create_hparams(args.hparams)
  training_dir_path = os.path.join(args.base_dir, args.training_dir)

  if not args.continue_training:
    use_map = args.weight_map_mode == 'use_map'
    map_path = os.path.join(training_dir_path, train_map_file)
    if use_map:
      log_train_map(training_dir_path, args.map)
    elif os.path.exists(map_path):
      os.remove(map_path)

    reset_log(training_dir_path)
    prepare_ms(args.base_dir, training_dir_path, speakers=args.speakers, pretrained_model=args.pretrained_model, weight_map_mode=args.weight_map_mode, hparams=hparams, pretrained_model_symbols=args.pretrained_model_symbols)
    split_ds(args.base_dir, training_dir_path, train_size=args.train_size, validation_size=args.validation_size, seed=args.seed, duration_col=duration_col)
    
  weights_path = os.path.join(get_filelist_dir(training_dir_path), filelist_weights_file_name)
  use_weights_map = os.path.exists(weights_path)
  start_train(training_dir_path, hparams=hparams, use_weights=use_weights_map, pretrained_path=args.pretrained_path, warm_start=args.warm_start, continue_training=args.continue_training)

  analyse(training_dir_path)