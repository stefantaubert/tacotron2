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

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--no_debugging', action='store_true')
  parser.add_argument('--base_dir', type=str, help='base directory')
  parser.add_argument('--training_dir', type=str)
  parser.add_argument('--ipa', action='store_true')
  parser.add_argument('--text', type=str)
  parser.add_argument('--is_ipa', action='store_true')
  parser.add_argument('--ignore_tones', action='store_true')
  parser.add_argument('--ignore_arcs', action='store_true')
  parser.add_argument('--map', type=str)
  parser.add_argument('--subset_id', type=str)
  parser.add_argument('--hparams', type=str)
  parser.add_argument('--waveglow', type=str)
  parser.add_argument('--custom_checkpoint', type=str)

  args = parser.parse_args()

  if not args.no_debugging:
    args.base_dir = '/datasets/models/taco2pt_v2'
    args.training_dir = 'debug'
    args.ipa = True
    args.text = "examples/ipa/north_sven_v2.txt"
    args.is_ipa = True
    args.ignore_tones = True
    args.ignore_arcs = True
    args.waveglow = "/datasets/models/pretrained/waveglow_256channels_universal_v5.pt"

  training_dir_path = os.path.join(args.base_dir, args.training_dir)

  assert os.path.isfile(args.text)
  assert os.path.isfile(args.waveglow)

  print("Infering text from:", args.text)
  input_name = os.path.splitext(os.path.basename(args.text))[0]
  if args.custom_checkpoint:
    checkpoint = args.custom_checkpoint
  else:
    checkpoint = get_last_checkpoint(training_dir_path)
  infer_dir_path = get_inference_dir(training_dir_path, checkpoint, input_name)
  log_inference_config(infer_dir_path, args)
  log_input_file(infer_dir_path, args.text)

  if args.map:
    assert os.path.isfile(args.map)
    print("Using mapping from:", args.map)
    log_map_file(infer_dir_path, args.map)
  else:
    print("Using no mapping.")

  process_input_text(training_dir_path, infer_dir_path, ipa=args.ipa, ignore_tones=args.ignore_tones, ignore_arcs=args.ignore_arcs, subset_id=args.subset_id, is_ipa=args.is_ipa, use_map=bool(args.map))
  infer(training_dir_path, infer_dir_path, hparams=args.hparams, waveglow=args.waveglow, custom_checkpoint=args.custom_checkpoint)
