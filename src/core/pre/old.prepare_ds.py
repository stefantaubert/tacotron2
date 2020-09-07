import argparse
import os
from math import sqrt
from shutil import copyfile

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn

from src.core.common import csv_separator, log, parse_json, symbols_str_col
from src.paths import (ds_preprocessed_file_name, ds_preprocessed_symbols_name,
                       filelist_file_log_name, filelist_file_name,
                       filelist_symbols_file_name, filelist_weights_file_name,
                       get_ds_dir, get_filelist_dir, train_map_file)
from src.tacotron.hparams import create_hparams
from src.text.symbol_id_dict import (deserialize_symbol_ids, load_from_file,
                                     serialize_symbol_ids)


def prepare(base_dir: str, training_dir_path: str, merge_mode: str, pretrained_model_symbols: str, ds_name: str, speaker: str, pretrained_model: str, weight_map_mode: str, hparams):
  log(training_dir_path, "Merging symbols...")
  speaker_dir_path = get_ds_dir(base_dir, ds_name, speaker)
  
  new_prepr_path = os.path.join(speaker_dir_path, ds_preprocessed_file_name)
  new_symbols_path = os.path.join(speaker_dir_path, ds_preprocessed_symbols_name)
  new_speaker_conv = load_from_file(new_symbols_path)

  new_data = pd.read_csv(new_prepr_path, header=None, sep=csv_separator)
  new_symbols = set(new_speaker_conv.get_symbols())
  # a_concat_b, a_union_b
  if merge_mode:
    assert pretrained_model_symbols
    final_conv = load_from_file(pretrained_model_symbols)
    if merge_mode == 'a_union_b':
      final_conv.add_symbols(new_symbols, ignore_existing=True, subset_id=1)
    elif merge_mode == 'a_concat_b':
      final_conv.add_symbols(new_symbols, ignore_existing=False, subset_id=1)
    else:
      raise Exception('merge_mode not supported:', merge_mode)

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

    # filelist.csv
    df = pd.DataFrame(result)
    print(df.head())
    df.to_csv(os.path.join(get_filelist_dir(training_dir_path), filelist_file_name), header=None, index=None, sep=csv_separator)
  else:
    # copy symbols.json
    a = os.path.join(speaker_dir_path, ds_preprocessed_symbols_name)
    b = os.path.join(get_filelist_dir(training_dir_path), filelist_symbols_file_name)
    copyfile(a, b)

    # copy filelist.csv
    a = os.path.join(speaker_dir_path, ds_preprocessed_file_name)
    b = os.path.join(get_filelist_dir(training_dir_path), filelist_file_name)
    copyfile(a, b)

    # # copy filelist_log.csv
    # a = os.path.join(speaker_dir_path, ds_preprocessed_file_log_name)
    # b = os.path.join(get_filelist_dir(training_dir_path), filelist_file_log_name)
    # copyfile(a, b)

  if weight_map_mode != None:
    assert pretrained_model
    assert pretrained_model_symbols
    pretrained_speaker_conv = load_from_file(pretrained_model_symbols)

    hparams = create_hparams(hparams)
    final_conv = load_from_file(os.path.join(get_filelist_dir(training_dir_path), filelist_symbols_file_name))
    n_symbols = final_conv.get_symbols_count()
    embedding = nn.Embedding(n_symbols, hparams.symbols_embedding_dim)
    std = sqrt(2.0 / (n_symbols + hparams.symbols_embedding_dim))
    val = sqrt(3.0) * std  # uniform bounds for std
    embedding.weight.data.uniform_(-val, val)

    checkpoint_dict = torch.load(pretrained_model, map_location='cpu')
    pretrained_emb = checkpoint_dict['state_dict']['embedding.weight']

    if weight_map_mode == 'same_symbols_only':
      a = set(pretrained_speaker_conv.get_symbols())
      b = set(final_conv.get_symbols())
      a_intersect_b = a.intersection(b)
      log(training_dir_path, "intersecting symbols {}".format(str(a_intersect_b)))
      ipa_mapping = { a: a for a in a_intersect_b }
    elif weight_map_mode == 'use_map':
      map_path = os.path.join(training_dir_path, train_map_file)
      ipa_mapping = parse_json(map_path)
    else:
      raise Exception('weight_map_mode not supported', weight_map_mode)
    
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
