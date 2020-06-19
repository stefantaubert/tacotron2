import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split
from torch import nn
import torch
from math import sqrt
import numpy as np
from utils import parse_map

from paths import get_filelist_dir, get_ds_dir, ds_preprocessed_file_name, ds_preprocessed_symbols_name, filelist_symbols_file_name, filelist_symbols_log_file_name, filelist_file_name, filelist_weights_file_name, train_map_file
from text.symbol_converter import load_from_file, serialize_symbol_ids, deserialize_symbol_ids
from utils import csv_separator
from hparams import create_hparams
from train_log import log

def merge_speakers(base_dir: str, training_dir_path: str, config: dict):
  log(training_dir_path, "Merging symbols...")
  ds_dir = get_ds_dir(base_dir, config["ds_name"], config["speaker"])
  
  new_prepr_path = os.path.join(ds_dir, ds_preprocessed_file_name)
  new_symbols_path = os.path.join(ds_dir, ds_preprocessed_symbols_name)
  
  pretrained_speaker_conv = load_from_file(config["pretrained_model_symbols"])
  new_speaker_conv = load_from_file(new_symbols_path)

  new_data = pd.read_csv(new_prepr_path, header=None, sep=csv_separator)
  pretrained_symbols_ids = pretrained_speaker_conv.get_symbol_ids()
  new_symbols = set(new_speaker_conv.get_symbols())

  if config["merge_mode"] == 'unify':
    pretrained_speaker_conv.add_symbols(new_symbols, ignore_existing=True, subset_id=1)
  elif config["merge_mode"] == 'separate':
    pretrained_speaker_conv.add_symbols(new_symbols, ignore_existing=False, subset_id=1)
  else:
    raise Exception('merge_mode not supported', config["merge_mode"])

  log(training_dir_path, "Resulting symbolset:")
  log(training_dir_path, '\n'.join(pretrained_speaker_conv.get_symbols(include_subset_id=True, include_id=True)))

  result = []
  for i, row in new_data.iterrows():
    serialized_ids = row[1]
    deserialized_ids = deserialize_symbol_ids(serialized_ids)
    original_symbols = new_speaker_conv.ids_to_symbols(deserialized_ids)
    updated_ids = pretrained_speaker_conv.symbols_to_ids(original_symbols, subset_id_if_multiple=1, add_eos=False, replace_unknown_with_pad=True)
    serialized_updated_ids = serialize_symbol_ids(updated_ids)
    row[1] = serialized_updated_ids
    result.append(row)

  pretrained_speaker_conv.dump(os.path.join(get_filelist_dir(training_dir_path), filelist_symbols_file_name))
  pretrained_speaker_conv.plot(os.path.join(get_filelist_dir(training_dir_path), filelist_symbols_log_file_name))
  df = pd.DataFrame(result)
  #print(df.head())
  df.to_csv(os.path.join(get_filelist_dir(training_dir_path), filelist_file_name), header=None, index=None, sep=csv_separator)

  hparams = create_hparams(config["hparams"])

  n_symbols = pretrained_speaker_conv.get_symbol_ids_count()
  embedding = nn.Embedding(n_symbols, hparams.symbols_embedding_dim)
  std = sqrt(2.0 / (n_symbols + hparams.symbols_embedding_dim))
  val = sqrt(3.0) * std  # uniform bounds for std
  embedding.weight.data.uniform_(-val, val)

  checkpoint_dict = torch.load(config["pretrained_model"], map_location='cpu')
  pretrained_emb = checkpoint_dict['state_dict']['embedding.weight']

  map_dataid_pretrainedid = []

  for symbol_id in pretrained_speaker_conv.get_symbol_ids():
    if symbol_id in pretrained_symbols_ids:
      map_dataid_pretrainedid.append((symbol_id, symbol_id))
      
  if config["map_pretrained_weights"]:
    # map: if destination occures multiple times, the last one is taken for initializing weights
    map_path = os.path.join(base_dir, train_map_file)
    ipa_mapping = parse_map(map_path)
    for source_symbol, dest_symbol in ipa_mapping.items():
      if dest_symbol == '':
        continue
      source_symbol_id = pretrained_speaker_conv.symbol_to_id(source_symbol, subset_id_if_multiple=0)
      dest_symbol = dest_symbol[0]
      dest_symbol_id = pretrained_speaker_conv.symbol_to_id(dest_symbol, subset_id_if_multiple=1)
      map_dataid_pretrainedid.append((dest_symbol_id, source_symbol_id))

  for data_id, pretrained_id in map_dataid_pretrainedid:
    pretrained_symbol = pretrained_speaker_conv.id_to_symbol(pretrained_id)
    dest_symbol = pretrained_speaker_conv.id_to_symbol(data_id)
    log(training_dir_path, 'Mapped pretrained weights from symbol {} to symbol {}'.format(pretrained_symbol, dest_symbol))
    embedding.weight.data[data_id] = pretrained_emb[pretrained_id]
  
  log(training_dir_path, str(embedding))

  weights_path = os.path.join(get_filelist_dir(training_dir_path), filelist_weights_file_name)
  np.save(weights_path, embedding.weight.data.numpy())

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

