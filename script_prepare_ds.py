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
from text.symbol_converter import load_from_file, serialize_symbol_ids, deserialize_symbol_ids
from utils import csv_separator
from hparams import create_hparams
from train_log import log
from shutil import copyfile
from utils import symbols_str_col

def prepare(base_dir: str, training_dir_path: str, config: dict):
  log(training_dir_path, "Merging symbols...")
  speaker_dir_path = get_ds_dir(base_dir, config["ds_name"], config["speaker"])
  
  new_prepr_path = os.path.join(speaker_dir_path, ds_preprocessed_file_name)
  new_symbols_path = os.path.join(speaker_dir_path, ds_preprocessed_symbols_name)
  new_speaker_conv = load_from_file(new_symbols_path)

  new_data = pd.read_csv(new_prepr_path, header=None, sep=csv_separator)
  new_symbols = set(new_speaker_conv.get_symbols())
  # b_only, a_concat_b, a_union_b
  if config["merge_mode"] == 'b_only':
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

    # # copy filelist_log.csv
    # a = os.path.join(speaker_dir_path, ds_preprocessed_file_log_name)
    # b = os.path.join(get_filelist_dir(training_dir_path), filelist_file_log_name)
    # copyfile(a, b)
  else:
    final_conv = load_from_file(config["pretrained_model_symbols"])
    if config["merge_mode"] == 'a_union_b':
      final_conv.add_symbols(new_symbols, ignore_existing=True, subset_id=1)
    elif config["merge_mode"] == 'a_concat_b':
      final_conv.add_symbols(new_symbols, ignore_existing=False, subset_id=1)
    else:
      raise Exception('merge_mode not supported', config["merge_mode"])

    log(training_dir_path, "Resulting symbolset:")
    log(training_dir_path, '\n'.join(final_conv.get_symbols(include_subset_id=True, include_id=True)))

    result = []
    for i, row in new_data.iterrows():
      serialized_ids = row[symbols_str_col]
      deserialized_ids = deserialize_symbol_ids(serialized_ids)
      original_symbols = new_speaker_conv.ids_to_symbols(deserialized_ids)
      updated_ids = final_conv.symbols_to_ids(original_symbols, subset_id_if_multiple=1, add_eos=False, replace_unknown_with_pad=True)
      serialized_updated_ids = serialize_symbol_ids(updated_ids)
      row[symbols_str_col] = serialized_updated_ids
      result.append(row)

    # symbols.json
    final_conv.dump(os.path.join(get_filelist_dir(training_dir_path), filelist_symbols_file_name))

    # symbols.log
    final_conv.plot(os.path.join(get_filelist_dir(training_dir_path), filelist_symbols_log_file_name))

    # filelist.csv
    df = pd.DataFrame(result)
    print(df.head())
    df.to_csv(os.path.join(get_filelist_dir(training_dir_path), filelist_file_name), header=None, index=None, sep=csv_separator)
  
  if config["weight_map_mode"] != 'none':
    pretrained_speaker_conv = load_from_file(config["pretrained_model_symbols"])

    hparams = create_hparams(config["hparams"])
    final_conv = load_from_file(os.path.join(get_filelist_dir(training_dir_path), filelist_symbols_file_name))
    n_symbols = final_conv.get_symbol_ids_count()
    embedding = nn.Embedding(n_symbols, hparams.symbols_embedding_dim)
    std = sqrt(2.0 / (n_symbols + hparams.symbols_embedding_dim))
    val = sqrt(3.0) * std  # uniform bounds for std
    embedding.weight.data.uniform_(-val, val)

    checkpoint_dict = torch.load(config["pretrained_model"], map_location='cpu')
    pretrained_emb = checkpoint_dict['state_dict']['embedding.weight']

    if config["weight_map_mode"] == 'same_symbols_only':
      a = set(pretrained_speaker_conv.get_symbols())
      b = set(final_conv.get_symbols())
      a_intersect_b = a.intersection(b)
      log(training_dir_path, "intersecting symbols {}".format(str(a_intersect_b)))
      ipa_mapping = { a: a for a in a_intersect_b }
    elif config["weight_map_mode"] == 'use_map':
      map_path = os.path.join(training_dir_path, train_map_file)
      ipa_mapping = parse_map_json(map_path)
    else:
      raise Exception('weight_map_mode not supported', config["weight_map_mode"])
    
    not_mapped = set()
    for final_symbol, source_symbol in ipa_mapping.items():
      if not final_conv.symbol_exists(final_symbol):
        log(training_dir_path, "Symbol {} doesn't exist in destination symbol set. Ignoring mapping from {}.".format(final_symbol, source_symbol))
        continue

      if not pretrained_speaker_conv.symbol_exists(source_symbol):
        log(training_dir_path, "Symbol {} doesn't exist in pretrained model. Ignoring mapping to {}.".format(source_symbol, final_symbol))
        not_mapped.add(final_symbol)
        continue

      source_symbol_id = pretrained_speaker_conv.symbol_to_id(source_symbol, subset_id_if_multiple=0)
      final_symbol_id = final_conv.symbol_to_id(final_symbol, subset_id_if_multiple=1)
      embedding.weight.data[final_symbol_id] = pretrained_emb[source_symbol_id]
      log(training_dir_path, 'Mapped pretrained weights from symbol {} ({}) to symbol {} ({})'.format(source_symbol, source_symbol_id, final_symbol, final_symbol_id))
    
    unmapped_symbols = set(final_conv.get_symbols()).difference(set(ipa_mapping.keys())).union(not_mapped)
    if len(unmapped_symbols) == 0:
      log(training_dir_path, "All symbols were mapped.")
    else:
      log(training_dir_path, "Symbols without initialized mapping: {}".format(str(unmapped_symbols)))

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
