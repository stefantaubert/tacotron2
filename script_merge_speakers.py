import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split
from torch import nn
import torch
from math import sqrt
import numpy as np

from paths import preprocessed_file_name, test_file_name, training_file_name, validation_file_name, filelist_dir, symbols_path_name, symbols_path_info_name, savecheckpoints_dir, weights_name
from text.symbol_converter import load_from_file, serialize_symbol_ids, deserialize_symbol_ids
from utils import csv_separator
from hparams import create_hparams

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('--base_dir', type=str, help='base directory')
  parser.add_argument('--pretrained_model', type=str)
  parser.add_argument('--pretrained_model_symbols', type=str)
  parser.add_argument('--ds_name', type=str)
  parser.add_argument('--speaker', type=str)
  parser.add_argument('--mode', type=str, help='separate,unify,map')

  args = parser.parse_args()
  debug = True
  if debug:
    args.base_dir = '/datasets/models/taco2pt_ms'
    args.pretrained_model = os.path.join(args.base_dir, savecheckpoints_dir, 'ljs_1_ipa_49000')
    args.pretrained_model_symbols = os.path.join(args.base_dir, filelist_dir, 'ljs_ipa/1/symbols.json')
    args.ds_name = 'thchs_no_tone'
    args.speaker = 'A11'
    args.mode = 'separate'

  filelist_dir_path = os.path.join(args.base_dir, filelist_dir)

  new_speaker_dir = os.path.join(filelist_dir_path, args.ds_name, args.speaker)
  new_prepr_path = os.path.join(new_speaker_dir, preprocessed_file_name)
  new_symbols_path = os.path.join(new_speaker_dir, symbols_path_name)
  
  pretrained_speaker_conv = load_from_file(args.pretrained_model_symbols)
  new_speaker_conv = load_from_file(new_symbols_path)

  new_data = pd.read_csv(new_prepr_path, header=None, sep=csv_separator)
  pretrained_symbols_ids = pretrained_speaker_conv.get_symbol_ids()
  new_symbols = set(new_speaker_conv.get_symbols())

  if args.mode == 'unify':
    pretrained_speaker_conv.add_symbols(new_symbols, ignore_existing=True, subset_id=1)
  elif args.mode == 'separate':
    pretrained_speaker_conv.add_symbols(new_symbols, ignore_existing=False, subset_id=1)

  print("Resulting symbolset:")
  print('\n'.join(pretrained_speaker_conv.get_symbols(include_subset_id=True, include_id=True)))

  result = []
  for i, row in new_data.iterrows():
    serialized_ids = row[1]
    deserialized_ids = deserialize_symbol_ids(serialized_ids)
    original_symbols = new_speaker_conv.ids_to_symbols(deserialized_ids)
    updated_ids = pretrained_speaker_conv.symbols_to_ids(original_symbols, subset_id_if_multiple=1, add_eos=False)
    serialized_updated_ids = serialize_symbol_ids(updated_ids)
    row[1] = serialized_updated_ids
    result.append(row)

  pretrained_speaker_conv.dump(os.path.join(filelist_dir_path, symbols_path_name))
  pretrained_speaker_conv.plot(os.path.join(filelist_dir_path, symbols_path_info_name))
  df = pd.DataFrame(result)
  print(df.head())
  df.to_csv(os.path.join(filelist_dir_path, preprocessed_file_name), header=None, index=None, sep=csv_separator)

  hparams = create_hparams('')

  n_symbols = pretrained_speaker_conv.get_symbol_ids_count()
  embedding = nn.Embedding(n_symbols, hparams.symbols_embedding_dim)
  std = sqrt(2.0 / (n_symbols + hparams.symbols_embedding_dim))
  val = sqrt(3.0) * std  # uniform bounds for std
  embedding.weight.data.uniform_(-val, val)

  checkpoint_dict = torch.load(args.pretrained_model, map_location='cpu')
  pretrained_emb = checkpoint_dict['state_dict']['embedding.weight']

  for symbol_id in pretrained_speaker_conv.get_symbol_ids():
    if symbol_id in pretrained_symbols_ids:
      embedding.weight.data[symbol_id] = pretrained_emb[symbol_id]
  
  print(embedding)

  weights_path = os.path.join(filelist_dir_path, weights_name)
  np.save(weights_path, embedding.weight.data.numpy())

  print("Done.")