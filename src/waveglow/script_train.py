import argparse
import json
import os
from shutil import copyfile

from src.waveglow.hparams import create_hparams
from src.script_paths import (ds_preprocessed_file_name,
                   ds_preprocessed_symbols_name, filelist_file_name,
                   filelist_symbols_file_name,
                   filelist_weights_file_name, get_ds_dir, get_filelist_dir,
                   inference_config_file,
                   log_inference_config, log_input_file, log_map_file,
                   log_train_config, log_train_map, train_config_file,
                   train_map_file)
from src.waveglow.prepare_ds import prepare, duration_col
from src.common.split_ds import split_ds
from src.waveglow.train import get_last_checkpoint, start_train
from src.common.train_log import reset_log
from src.common.utils import args_to_str

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
    args.hparams = 'batch_size=4,iters_per_checkpoint=5,fp16_run=False,with_tensorboard=True'
    args.continue_training = True

  if not args.base_dir:
    raise Exception("Argument 'base_dir' is required.")
  elif not args.training_dir:
    raise Exception("Argument 'training_dir' is required.")

  hparams = create_hparams(args.hparams)
  training_dir_path = os.path.join(args.base_dir, args.training_dir)

  if not args.continue_training:
    reset_log(training_dir_path)

    prepare(
      base_dir=args.base_dir,
      training_dir_path=training_dir_path,
      speakers=args.speakers
    )

    split_ds(
      base_dir=args.base_dir,
      training_dir_path=training_dir_path,
      train_size=args.train_size,
      validation_size=args.validation_size,
      seed=args.seed,
      duration_col=duration_col
    )
    
  start_train(
    training_dir_path=training_dir_path,
    hparams=hparams,
    continue_training=args.continue_training
  )
