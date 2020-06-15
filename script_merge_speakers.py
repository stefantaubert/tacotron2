import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split
from torch import nn
import torch
from math import sqrt
import numpy as np

from paths import preprocessed_file_name, test_file_name, training_file_name, validation_file_name, filelist_dir, symbols_path_name, symbols_path_info_name, savecheckpoints_dir, weights_name
from text.conversion.SymbolConverter import get_from_file, get_from_symbols, _eos, _pad, get_symbols_from_str
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
    args.base_dir = '/datasets/models/taco2pt_ms'
    args.ds_name = 'thchs_no_tone'
    args.speaker = 'A11'
    args.mode = 'unify'

  filelist_dir_path = os.path.join(args.base_dir, filelist_dir)

  new_speaker_dir = os.path.join(filelist_dir_path, args.ds_name, args.speaker)
  new_prepr_path = os.path.join(new_speaker_dir, preprocessed_file_name)
  new_symbols_path = os.path.join(new_speaker_dir, symbols_path_name)
  
  pretrained_speaker_conv = get_from_file(args.pretrained_model_symbols)
  new_speaker_conv = get_from_file(new_symbols_path)

  new_data = pd.read_csv(new_prepr_path, header=None, sep=csv_separator)

  if args.mode == 'unify':
    existing_symbols = pretrained_speaker_conv.get_symbols().intersection(new_speaker_conv.get_symbols())
    new_symbols = new_speaker_conv.get_symbols().difference(pretrained_speaker_conv.get_symbols())
    pretrained_speaker_conv.add_symbols(new_symbols)

  result = []
  for i, row in new_data.iterrows():
    symb_seq = row[1]
    symb_seq_int = get_symbols_from_str(symb_seq)
    original_symbols = new_speaker_conv.sequence_to_original_chars(symb_seq_int)
    new_symb_seq = pretrained_speaker_conv.text_to_sequence(original_symbols)
    seq_str = ",".join([str(s) for s in new_symb_seq])
    row[1] = seq_str
    result.append(row)

  pretrained_speaker_conv.dump(os.path.join(filelist_dir_path, symbols_path_name))
  pretrained_speaker_conv.plot(os.path.join(filelist_dir_path, symbols_path_info_name), sort=False)
  df = pd.DataFrame(result)
  print(df.head())
  df.to_csv(os.path.join(filelist_dir_path, preprocessed_file_name), header=None, index=None, sep=csv_separator)

  hparams = create_hparams('')

  n_symbols = pretrained_speaker_conv.get_symbols_count()
  embedding = nn.Embedding(n_symbols, hparams.symbols_embedding_dim)
  std = sqrt(2.0 / (n_symbols + hparams.symbols_embedding_dim))
  val = sqrt(3.0) * std  # uniform bounds for std
  embedding.weight.data.uniform_(-val, val)

  checkpoint_dict = torch.load(args.pretrained_model, map_location='cpu')
  pretrained_emb = checkpoint_dict['state_dict']['embedding.weight']

  for symbol, symbol_id in pretrained_speaker_conv._id_to_symbol.items():
    if symbol in existing_symbols:
      embedding.weight.data[symbol_id] = pretrained_emb[symbol_id]
  
  print(embedding)

  weights_path = os.path.join(args.base_dir, weights_name)
  np.save(weights_path, embedding.weight.data.numpy())

  print("Done.")