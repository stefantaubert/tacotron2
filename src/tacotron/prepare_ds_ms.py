import argparse
import os
from collections import OrderedDict
from math import sqrt
from shutil import copyfile

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch
from src.common.train_log import log
from src.common.utils import (csv_separator, duration_col, parse_ds_speakers,
                              parse_json, save_json, serialize_ds_speaker,
                              serialize_ds_speakers, speaker_id_col,
                              speaker_name_col, symbols_str_col, utt_name_col,
                              wavpath_col, str_to_int)
from src.script_paths import (ds_preprocessed_file_name,
                              ds_preprocessed_symbols_name,
                              filelist_file_log_name, filelist_file_name,
                              filelist_speakers_name,
                              filelist_symbols_file_name,
                              filelist_test_file_name,
                              filelist_training_file_name,
                              filelist_validation_file_name,
                              filelist_weights_file_name,
                              get_all_speakers_path, get_ds_dir,
                              get_filelist_dir, train_map_file)
from src.tacotron.hparams import create_hparams
from src.text.symbol_converter import (deserialize_symbol_ids,
                                       init_from_symbols, load_from_file,
                                       serialize_symbol_ids)
from torch import nn
from src.common.utils import csv_separator, get_total_duration_min


def prepare(base_dir: str, training_dir_path: str, speakers: str, pretrained_model_symbols: str, pretrained_model: str, weight_map_mode: str, hparams, test_size: float, val_size: float, seed: int):
  ds_speakers = parse_ds_speakers(speakers)
  final_conv = init_from_symbols(set())
  
  # expand all
  expanded_speakers = []
  for ds, speaker, _ in ds_speakers:
    if speaker == 'all':
      all_speakers_path = get_all_speakers_path(base_dir, ds)
      all_speakers = parse_json(all_speakers_path)
      all_speakers = sorted(all_speakers.keys())
      for speaker_name in all_speakers:
        expanded_speakers.append((ds, speaker_name))
    else:
      expanded_speakers.append((ds, speaker))
  
  expanded_speakers = list(sorted(set(expanded_speakers)))
  new_speakers_str = serialize_ds_speakers(expanded_speakers)
  ds_speakers = parse_ds_speakers(new_speakers_str)
  speakers_info = OrderedDict([(serialize_ds_speaker(ds, speaker), s_id) for ds, speaker, s_id in ds_speakers])
  speakers_file = os.path.join(get_filelist_dir(training_dir_path), filelist_speakers_name)
  save_json(speakers_file, speakers_info)
  print(speakers_info)

  for ds, speaker, _ in ds_speakers:
    speaker_dir_path = get_ds_dir(base_dir, ds, speaker)
    symbols_path = os.path.join(speaker_dir_path, ds_preprocessed_symbols_name)
    speaker_conv = load_from_file(symbols_path)
    speaker_symbols = set(speaker_conv.get_symbols(include_id=False, include_subset_id=False))
    final_conv.add_symbols(speaker_symbols, ignore_existing=True, subset_id=0)

  # symbols.json
  final_conv.dump(os.path.join(get_filelist_dir(training_dir_path), filelist_symbols_file_name))

  wholeset = []
  testset = []
  trainset = []
  valset = []

  create_testset = test_size > 0
  create_valset = val_size > 0

  for ds, speaker, speaker_id in tqdm(ds_speakers):
    speaker_dir_path = get_ds_dir(base_dir, ds, speaker)
    prepr_path = os.path.join(speaker_dir_path, ds_preprocessed_file_name)
    symbols_path = os.path.join(speaker_dir_path, ds_preprocessed_symbols_name)
    speaker_conv = load_from_file(symbols_path)
    speaker_data = pd.read_csv(prepr_path, header=None, sep=csv_separator)

    speaker_new_rows = []

    for _, row in speaker_data.iterrows():
      serialized_ids = row[symbols_str_col]
      deserialized_ids = deserialize_symbol_ids(serialized_ids)
      original_symbols = speaker_conv.ids_to_symbols(deserialized_ids)
      updated_ids = final_conv.symbols_to_ids(original_symbols, subset_id_if_multiple=1, add_eos=False, replace_unknown_with_pad=True)
      serialized_updated_ids = serialize_symbol_ids(updated_ids)
      basename = row[0]
      wav_path = row[1]
      duration = row[3]
      new_row = [''] * 6
      new_row[utt_name_col] = basename
      new_row[wavpath_col] = wav_path
      new_row[symbols_str_col] = serialized_updated_ids
      new_row[duration_col] = duration
      new_row[speaker_id_col] = speaker_id
      new_row[speaker_name_col] = speaker
      #new_row = [basename, wav_path, serialized_updated_ids, duration, speaker_id, speaker]
      speaker_new_rows.append(new_row)
    
    # reason: speakers with same utterance counts should not have the same validation sets
    speaker_seed = seed + str_to_int(speaker)

    train = speaker_new_rows
    if create_testset:
      train, test = train_test_split(train, test_size=test_size, random_state=speaker_seed)
      testset.extend(test)

    if create_valset:
      train, val = train_test_split(train, test_size=val_size, random_state=speaker_seed)
      valset.extend(val)

    trainset.extend(train)
    wholeset.extend(speaker_new_rows)


  # filelist.csv
  trainset_path = os.path.join(get_filelist_dir(training_dir_path), filelist_training_file_name)
  df = pd.DataFrame(trainset)
  df.to_csv(trainset_path, header=None, index=None, sep=csv_separator)
  total_dur_min = get_total_duration_min(df, duration_col)
  log(training_dir_path, "{} => Size: {}, Duration: {:.2f}min / {:.2f}h".format(filelist_training_file_name, len(df), total_dur_min, total_dur_min / 60))

  if create_testset:
    testset_path = os.path.join(get_filelist_dir(training_dir_path), filelist_test_file_name)
    df = pd.DataFrame(testset)
    df.to_csv(testset_path, header=None, index=None, sep=csv_separator)
    total_dur_min = get_total_duration_min(df, duration_col)
    log(training_dir_path, "{} => Size: {}, Duration: {:.2f}min / {:.2f}h".format(filelist_test_file_name, len(df), total_dur_min, total_dur_min / 60))
  else:
    log(training_dir_path, "Create no testset.")

  if create_valset:
    valset_path = os.path.join(get_filelist_dir(training_dir_path), filelist_validation_file_name)
    df = pd.DataFrame(valset)
    df.to_csv(valset_path, header=None, index=None, sep=csv_separator)
    total_dur_min = get_total_duration_min(df, duration_col)
    log(training_dir_path, "{} => Size: {}, Duration: {:.2f}min / {:.2f}h".format(filelist_validation_file_name, len(df), total_dur_min, total_dur_min / 60))
  else:
    log(training_dir_path, "Create no valset.")
  
  # filelist.csv
  wholeset_path = os.path.join(get_filelist_dir(training_dir_path), filelist_file_name)
  df = pd.DataFrame(wholeset)
  df.to_csv(wholeset_path, header=None, index=None, sep=csv_separator)
  total_dur_min = get_total_duration_min(df, duration_col)
  log(training_dir_path, "{} => Size: {}, Duration: {:.2f}min / {:.2f}h".format(filelist_file_name, len(df), total_dur_min, total_dur_min / 60))
  print(df.head())


  weights_path = os.path.join(get_filelist_dir(training_dir_path), filelist_weights_file_name)

  if weight_map_mode == None:
    if os.path.exists(weights_path):
      os.remove(weights_path)
  else:
    assert pretrained_model
    assert pretrained_model_symbols
    pretrained_speaker_conv = load_from_file(pretrained_model_symbols)

    n_symbols = final_conv.get_symbol_ids_count()
    embedding = nn.Embedding(n_symbols, hparams.symbols_embedding_dim)
    std = sqrt(2.0 / (n_symbols + hparams.symbols_embedding_dim))
    val = sqrt(3.0) * std  # uniform bounds for std
    embedding.weight.data.uniform_(-val, val)

    checkpoint_dict = torch.load(pretrained_model, map_location='cpu')
    pretrained_emb = checkpoint_dict['state_dict']['embedding.weight']

    symbols_match_model = len(pretrained_emb) == pretrained_speaker_conv.get_symbol_ids_count()
    if not symbols_match_model:
      error_msg = "Weights mapping: symbol space from pretrained model ({}) did not match amount of symbols ({}).".format(len(pretrained_emb), pretrained_speaker_conv.get_symbol_ids_count())
      raise Exception(error_msg)

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
    np.save(weights_path, embedding.weight.data.numpy())

  log(training_dir_path, "Done.")
