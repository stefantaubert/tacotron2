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

def prepare(base_dir: str, training_dir_path: str, speakers: str, pretrained_model_symbols: str, pretrained_model: str, weight_map_mode: str, hparams):
  ds_speakers = parse_ds_speakers(speakers)
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

  if weight_map_mode != None:
    assert pretrained_model
    assert pretrained_model_symbols
    pretrained_speaker_conv = load_from_file(pretrained_model_symbols)

    hp = create_hparams(hparams)
    n_symbols = final_conv.get_symbol_ids_count()
    embedding = nn.Embedding(n_symbols, hp.symbols_embedding_dim)
    std = sqrt(2.0 / (n_symbols + hp.symbols_embedding_dim))
    val = sqrt(3.0) * std  # uniform bounds for std
    embedding.weight.data.uniform_(-val, val)

    checkpoint_dict = torch.load(pretrained_model, map_location='cpu')
    pretrained_emb = checkpoint_dict['state_dict']['embedding.weight']

    symbols_match_model = len(pretrained_emb) == pretrained_speaker_conv.get_symbol_ids_count()
    if not symbols_match_model:
      error_msg = "Weights mapping: symbol space from pretrained model ({}) did not match size of symbols ({})".format(len(pretrained_emb), pretrained_speaker_conv.get_symbol_ids_count())
      raise Exception(error_msg)

    if weight_map_mode == 'same_symbols_only':
      a = set(pretrained_speaker_conv.get_symbols())
      b = set(final_conv.get_symbols())
      a_intersect_b = a.intersection(b)
      log(training_dir_path, "intersecting symbols {}".format(str(a_intersect_b)))
      ipa_mapping = { a: a for a in a_intersect_b }
    elif weight_map_mode == 'use_map':
      map_path = os.path.join(training_dir_path, train_map_file)
      ipa_mapping = parse_map_json(map_path)
    else:
      raise Exception('weight_map_mode not supported {}'.format(weight_map_mode))
    
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
