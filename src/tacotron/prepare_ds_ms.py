import argparse
import os
from collections import OrderedDict
from math import sqrt
from shutil import copyfile

import numpy as np
from tqdm import tqdm

import torch
from src.common.train_log import log
from src.common.utils import (parse_ds_speakers, parse_json, str_to_int,
                              serialize_ds_speaker, serialize_ds_speakers)
from src.pre.text_pre_io import TextData, TextDataList, parse_data, parse_symbols, expand_speakers
from src.tacotron.prepare_ds_ms_io import (remove_weights_file,
                                           save_all_speakers, save_all_symbols, remove_train_map,
                                           save_testset, save_trainset, save_train_map, save_train_map, parse_weights_map,
                                           save_validationset, save_weights,
                                           save_wholeset, PreparedData, PreparedDataList)
from src.text.symbol_converter import (deserialize_symbol_ids,
                                       init_from_symbols, load_from_file,
                                       serialize_symbol_ids)
from torch import nn
from src.common.utils import split_train_test_val

def expand_speakers_core(base_dir, speakers):
  # expand all
  ds_speakers = parse_ds_speakers(speakers)
  expanded_speakers = expand_speakers(base_dir, ds_speakers)
  new_speakers_str = serialize_ds_speakers(expanded_speakers)
  return new_speakers_str

def prepare(base_dir: str, training_dir_path: str, speakers: str, weight_map, pretrained_model_symbols: str, pretrained_model: str, weight_map_mode: str, hparams, test_size: float, val_size: float, seed: int):
  use_map = weight_map_mode == 'use_map'
  if use_map:
    save_train_map(training_dir_path, weight_map)
  else:
    remove_train_map(training_dir_path)

  final_conv = init_from_symbols(set())
  
  # expand all
  expanded_speakers = expand_speakers_core(base_dir, speakers)
  ds_speakers = parse_ds_speakers(expanded_speakers)
  speakers_info = OrderedDict([(serialize_ds_speaker(ds, speaker_name), s_id) for ds, speaker_name, s_id in ds_speakers])
  save_all_speakers(training_dir_path, speakers_info)

  for ds, speaker_name, _ in ds_speakers:
    speaker_conv = parse_symbols(base_dir, ds, speaker_name)
    speaker_symbols = set(speaker_conv.get_symbols(include_id=False, include_subset_id=False))
    final_conv.add_symbols(speaker_symbols, ignore_existing=True, subset_id=0)

  save_all_symbols(training_dir_path, final_conv)

  wholeset: PreparedDataList = []
  testset: PreparedDataList = []
  trainset: PreparedDataList = []
  valset: PreparedDataList = []

  print("Reading symbols...")
  for ds, speaker_name, speaker_id in tqdm(ds_speakers):
    speaker_conv = parse_symbols(base_dir, ds, speaker_name)
    speaker_data = parse_data(base_dir, ds, speaker_name)

    speaker_new_rows: PreparedDataList = []

    values: TextData
    for values in speaker_data:
      deserialized_ids = deserialize_symbol_ids(values.serialized_symbol_ids)
      original_symbols = speaker_conv.ids_to_symbols(deserialized_ids)
      updated_ids = final_conv.symbols_to_ids(original_symbols, subset_id_if_multiple=1, add_eos=False, replace_unknown_with_pad=True)
      serialized_updated_ids = serialize_symbol_ids(updated_ids)
      
      prepared_data = PreparedData(values.i, values.basename, values.wav_path, values.mel_path, serialized_updated_ids, values.duration, speaker_id, speaker_name)
      speaker_new_rows.append(prepared_data)
    
    # reason: speakers with same utterance counts should not have the same validation sets
    speaker_seed = seed + str_to_int(speaker_name)
    speaker_train, speaker_test, speaker_val = split_train_test_val(speaker_new_rows, test_size, val_size, speaker_seed)
    wholeset.extend(speaker_new_rows)
    trainset.extend(speaker_train)
    testset.extend(speaker_test)
    valset.extend(speaker_val)

  print("Saving datasets...")
  # filelist.csv
  save_trainset(training_dir_path, trainset)
 
  # filelist.csv
  save_wholeset(training_dir_path, wholeset)

  if len(testset) > 0:
    save_testset(training_dir_path, testset)
  else:
    log(training_dir_path, "Create no testset.")

  if len(valset) > 0:
    save_validationset(training_dir_path, valset)
  else:
    log(training_dir_path, "Create no valset.")
  
  if weight_map_mode == None:
    remove_weights_file(training_dir_path)
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
      ipa_mapping = parse_weights_map(training_dir_path)
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
    save_weights(training_dir_path, embedding)

  log(training_dir_path, "Done.")

if __name__ == "__main__":
  prepare(
    base_dir="/datasets/models/taco2pt_v2",
    training_dir_path="/datasets/models/taco2pt_v2/debug",
    speakers="thchs_mel_v1,all",
    pretrained_model_symbols=None,
    pretrained_model=None,
    weight_map_mode=None,
    hparams=None,
    test_size=0.001,
    val_size=0.01,
    seed=1234,
    weight_map=None
  )
