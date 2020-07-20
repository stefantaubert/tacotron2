import argparse
import json
import os
import random
from shutil import copyfile
import pandas as pd
from utils import csv_separator, utt_name_col, symbols_str_col, wavpath_col, speaker_id_col, speaker_name_col, get_speaker_count_csv, get_utterance_names_csv
from paths import (ds_preprocessed_file_name, filelist_validation_file_name,
                   ds_preprocessed_symbols_name, filelist_file_name,
                   filelist_symbols_file_name,
                   get_ds_dir, get_filelist_dir, get_inference_dir, get_validation_dir,
                   inference_config_file, log_inference_config, log_input_file,
                   log_map_file, log_train_config, train_config_file, log_train_map)
from script_prepare_ds import prepare
from script_split_ds import split_ds
from script_txt_pre import process_input_text
from synthesize import validate
from train import start_train, get_last_checkpoint
from train_log import reset_log
from plot_embeddings import analyse
from utils import parse_ds_speaker

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--no_debugging', action='store_true')
  parser.add_argument('--base_dir', type=str, help='base directory')
  parser.add_argument('--training_dir', type=str)
  parser.add_argument('--utterance', type=str, help="Utterance name or random-val")
  parser.add_argument('--hparams', type=str)
  parser.add_argument('--waveglow', type=str)
  parser.add_argument('--custom_checkpoint', type=str)

  args = parser.parse_args()

  if not args.no_debugging:
    args.base_dir = '/datasets/models/taco2pt_v2'
    args.training_dir = 'debug_thchs_ms_v2'
    #args.utterance = "LJ001-0001"
    args.utterance = "random-val"
    args.waveglow = "/datasets/models/pretrained/waveglow_256channels_universal_v5.pt"

  training_dir_path = os.path.join(args.base_dir, args.training_dir)

  assert os.path.isfile(args.waveglow)

  if args.custom_checkpoint:
    checkpoint = args.custom_checkpoint
  else:
    checkpoint = get_last_checkpoint(training_dir_path)

  preprocessed_path = os.path.join(get_filelist_dir(training_dir_path), ds_preprocessed_file_name)

  infer_utterance_name = args.utterance

  if infer_utterance_name == "random-val":
    valset_path = os.path.join(get_filelist_dir(training_dir_path), filelist_validation_file_name)
    valset = pd.read_csv(valset_path, header=None, sep=csv_separator)
    all_names = get_utterance_names_csv(valset)
    infer_utterance_name = random.choice(list(all_names))
    print("Selected random validationset utterance: {}".format(infer_utterance_name))

  data = pd.read_csv(preprocessed_path, header=None, sep=csv_separator)
  infer_data = None
  for i, row in data.iterrows():
    utt_name = row[utt_name_col]
    if utt_name == infer_utterance_name:
      symbs = row[symbols_str_col]
      wav_path = row[wavpath_col]
      speaker_id = row[speaker_id_col]
      speaker_name = row[speaker_name_col]
      infer_data = (utt_name, symbs, wav_path, speaker_id)
      break

  if not infer_data:
    raise Exception("Utterance {} was not found".format(infer_utterance_name))
  
  print("Speaker is: {} ({})".format(speaker_name, str(speaker_id)))
  infer_dir_path = get_validation_dir(training_dir_path, infer_utterance_name, checkpoint, speaker_name)

  validate(training_dir_path, infer_dir_path, hparams=args.hparams, waveglow=args.waveglow, checkpoint=checkpoint, infer_data=infer_data)
