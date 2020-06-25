import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split
from torch import nn
import torch
from math import sqrt
import numpy as np
from utils import parse_map_json

from paths import get_filelist_dir, get_ds_dir, ds_preprocessed_file_name, ds_preprocessed_symbols_name, filelist_symbols_file_name, filelist_symbols_log_file_name, filelist_file_name, filelist_weights_file_name, train_map_file, ds_preprocessed_symbols_log_name, filelist_file_log_name
from text.symbol_converter import load_from_file, serialize_symbol_ids, deserialize_symbol_ids, init_from_symbols
from utils import csv_separator
from hparams import create_hparams
from train_log import log
from shutil import copyfile
from utils import symbols_str_col, parse_ds_speakers

def prepare(base_dir: str, training_dir_path: str, config: dict):
  ds_speakers = parse_ds_speakers(args.speakers)
  final_conv = init_from_symbols(set())
  
  for ds, speaker, _ in ds_speakers:
    speaker_dir_path = get_ds_dir(base_dir, ds, speaker)
    symbols_path = os.path.join(speaker_dir_path, ds_preprocessed_symbols_name)
    speaker_conv = load_from_file(symbols_path)
    speaker_symbols = set(speaker_conv.get_symbols(include_id=False, include_subset_id=False))
    final_conv.add_symbols(speaker_symbols, ignore_existing=True, subset_id=0)

  # symbols.json
  final_conv.dump(os.path.join(get_filelist_dir(training_dir_path), filelist_symbols_file_name))

  # symbols.log
  final_conv.plot(os.path.join(get_filelist_dir(training_dir_path), filelist_symbols_log_file_name))

  result = []

  for ds, speaker, speaker_id in ds_speakers:
    speaker_dir_path = get_ds_dir(base_dir, ds, speaker)
    prepr_path = os.path.join(speaker_dir_path, ds_preprocessed_file_name)
    symbols_path = os.path.join(speaker_dir_path, ds_preprocessed_symbols_name)
    speaker_conv = load_from_file(symbols_path)
    speaker_data = pd.read_csv(prepr_path, header=None, sep=csv_separator)

    for _, row in speaker_data.iterrows():
      serialized_ids = row[symbols_str_col]
      deserialized_ids = deserialize_symbol_ids(serialized_ids)
      original_symbols = speaker_conv.ids_to_symbols(deserialized_ids)
      updated_ids = final_conv.symbols_to_ids(original_symbols, subset_id_if_multiple=1, add_eos=False, replace_unknown_with_pad=True)
      serialized_updated_ids = serialize_symbol_ids(updated_ids)
      basename = row[0]
      wav_path = row[1]
      duration = row[3]
      new_row = [basename, wav_path, serialized_updated_ids, duration, speaker_id, speaker]
      result.append(new_row)

  # filelist.csv
  df = pd.DataFrame(result)
  print(df.head())
  df.to_csv(os.path.join(get_filelist_dir(training_dir_path), filelist_file_name), header=None, index=None, sep=csv_separator)

  log(training_dir_path, "Done.")

# if __name__ == "__main__":

#   parser = argparse.ArgumentParser()
#   parser.add_argument('--base_dir', type=str, help='base directory')
#   parser.add_argument('--pretrained_model', type=str)
#   parser.add_argument('--pretrained_model_symbols', type=str)
#   parser.add_argument('--ds_name', type=str)
#   parser.add_argument('--speaker', type=str)
#   parser.add_argument('--mode', type=str, help='separate,unify,map')
#   parser.add_argument('--map', type=str)

#   args = parser.parse_args()
#   debug = True
#   if debug:
#     args.base_dir = '/datasets/models/taco2pt_ms'
#     args.pretrained_model = os.path.join(args.base_dir, savecheckpoints_dir, 'ljs_1_ipa_49000')
#     args.pretrained_model_symbols = os.path.join(args.base_dir, filelist_dir, 'ljs_ipa/1/symbols.json')
#     args.ds_name = 'thchs_no_tone'
#     args.speaker = 'A11'
#     args.mode = 'map'
#     args.map = 'maps/en_chn.txt'

