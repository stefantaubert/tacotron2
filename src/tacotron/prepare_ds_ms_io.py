import random
import os
from collections import OrderedDict

import numpy as np

from src.common.train_log import log
from src.common.utils import (load_csv, parse_json,
                              save_csv, save_json)
from src.paths import (ds_preprocessed_file_name, ds_preprocessed_symbols_name,
                       filelist_file_log_name, filelist_file_name,
                       filelist_speakers_name, filelist_symbols_file_name,
                       filelist_test_file_name, filelist_training_file_name,
                       filelist_validation_file_name,
                       filelist_weights_file_name, get_all_speakers_path,
                       get_ds_dir, get_filelist_dir, train_map_file)
from src.text.symbol_converter import load_from_file

__id_idx = 0
__basename_idx = 1
__wav_path_idx = 2
__mel_path_idx = 3
__serialized_ids_idx = 4
__duration_idx = 5
__speaker_id_idx = 6
__speaker_name_idx = 7
__count = 8

def get_id(values):
  return values[__id_idx]

def get_basename(values):
  return values[__basename_idx]

def get_speaker_id(values):
  return values[__speaker_id_idx]

def get_speaker_name(values):
  return values[__speaker_name_idx]

def get_serialized_ids(values):
  return values[__serialized_ids_idx]

def get_duration(values):
  return values[__duration_idx]

def get_wav_path(values):
  return values[__wav_path_idx]

def get_mel_path(values):
  return values[__mel_path_idx]

def to_values(i, basename, wav_path, mel_path, serialized_updated_ids, duration, speaker_id, speaker_name):
  res = [''] * __count
  res[__id_idx] = i
  res[__basename_idx] = basename
  res[__wav_path_idx] = wav_path
  res[__mel_path_idx] = mel_path
  res[__serialized_ids_idx] = serialized_updated_ids
  res[__duration_idx] = duration
  res[__speaker_id_idx] = speaker_id
  res[__speaker_name_idx] = speaker_name
  return res

def get_symbols_path(training_dir_path: str) -> str:
  path = os.path.join(get_filelist_dir(training_dir_path), filelist_symbols_file_name)
  return path

def save_all_symbols(training_dir_path: str, conv):
  # symbols.json
  sym_path = get_symbols_path(training_dir_path)
  conv.dump(sym_path)

def parse_all_symbols(training_dir_path: str):
  sym_path = get_symbols_path(training_dir_path)
  conv = load_from_file(sym_path)
  return conv
  
def get_speakers_path(training_dir_path: str) -> str:
  path = os.path.join(get_filelist_dir(training_dir_path), filelist_speakers_name)
  return path

def save_all_speakers(training_dir_path: str, speakers_info: OrderedDict):
  speakers_file = get_speakers_path(training_dir_path)
  save_json(speakers_file, speakers_info)

def parse_all_speakers(training_dir_path: str) -> dict:
  speakers_file = get_speakers_path(training_dir_path)
  all_speakers = parse_json(speakers_file)
  return all_speakers

def get_speaker_id_from_name(training_dir_path: str, speaker_name: str):
  final_speaker_id = -1
  all_speakers = parse_all_speakers(training_dir_path)
  for ds_speaker, speaker_id in all_speakers.items():
    if ds_speaker == speaker_name:
      final_speaker_id = speaker_id
      break
    
  if final_speaker_id == -1:
    raise Exception("Speaker {} not available!".format(speaker_name))

  return final_speaker_id

def save_trainset(training_dir_path, dataset: list):
  dest_path = os.path.join(get_filelist_dir(training_dir_path), filelist_training_file_name)
  df = save_csv(dataset, dest_path)

  total_dur_min = get_total_duration(df.values) / 60
  log(training_dir_path, "{} => Size: {}, Duration: {:.2f}min / {:.2f}h".format(filelist_training_file_name, len(df), total_dur_min, total_dur_min / 60))

def get_total_duration(values):
  durations = [get_duration(x) for x in values]
  duration = sum(durations)
  return duration
  
def save_testset(training_dir_path, dataset: list):
  dest_path = os.path.join(get_filelist_dir(training_dir_path), filelist_test_file_name)
  df = save_csv(dataset, dest_path)
  total_dur_min = get_total_duration(df.values) / 60
  log(training_dir_path, "{} => Size: {}, Duration: {:.2f}min / {:.2f}h".format(filelist_test_file_name, len(df), total_dur_min, total_dur_min / 60))

def save_validationset(training_dir_path, dataset: list):
  dest_path = os.path.join(get_filelist_dir(training_dir_path), filelist_validation_file_name)
  df = save_csv(dataset, dest_path)
  total_dur_min = get_total_duration(df.values) / 60
  log(training_dir_path, "{} => Size: {}, Duration: {:.2f}min / {:.2f}h".format(filelist_validation_file_name, len(df), total_dur_min, total_dur_min / 60))

def save_wholeset(training_dir_path, dataset: list):
  dest_path = os.path.join(get_filelist_dir(training_dir_path), filelist_file_name)
  df = save_csv(dataset, dest_path)
  total_dur_min = get_total_duration(df.values) / 60
  log(training_dir_path, "{} => Size: {}, Duration: {:.2f}min / {:.2f}h".format(filelist_file_name, len(df), total_dur_min, total_dur_min / 60))

def parse_traindata(training_dir_path: str):
  dest_path = os.path.join(get_filelist_dir(training_dir_path), filelist_training_file_name)
  return load_csv(dest_path).values

def parse_validationset(training_dir_path: str):
  dest_path = os.path.join(get_filelist_dir(training_dir_path), filelist_validation_file_name)
  return load_csv(dest_path).values

def parse_testset(training_dir_path: str):
  dest_path = os.path.join(get_filelist_dir(training_dir_path), filelist_test_file_name)
  return load_csv(dest_path).values

def parse_wholeset(training_dir_path: str):
  dest_path = os.path.join(get_filelist_dir(training_dir_path), filelist_file_name)
  return load_csv(dest_path).values

def remove_weights_file(training_dir_path: str):
  weights_path = os.path.join(get_filelist_dir(training_dir_path), filelist_weights_file_name)
  if os.path.exists(weights_path):
    os.remove(weights_path)

def parse_weights_map(training_dir_path: str):
  map_path = os.path.join(training_dir_path, train_map_file)
  ipa_mapping = parse_json(map_path)
  return ipa_mapping

def save_weights(training_dir_path: str, embedding):
  weights_path = os.path.join(get_filelist_dir(training_dir_path), filelist_weights_file_name)
  np.save(weights_path, embedding.weight.data.numpy())

def get_random_val_utterance(training_dir_path: str, custom_speaker: str):
  dataset = parse_testset(training_dir_path)
  return __get_random_utterance(dataset, custom_speaker)

def get_random_test_utterance(training_dir_path: str, custom_speaker: str):
  dataset = parse_testset(training_dir_path)
  return __get_random_utterance(dataset, custom_speaker)

def __get_random_utterance(dataset, custom_speaker: str):
  random_value = random.choice(dataset)
  speaker_name = get_speaker_name(random_value)
  if custom_speaker:
    while True:
      if speaker_name == custom_speaker:
        break
      random_value = random.choice(dataset)
      speaker_name = get_speaker_name(random_value)
  return random_value

def get_values_entry(training_dir_path, dest_id: int):
  wholeset = parse_wholeset(training_dir_path)
  for values in wholeset:
    if get_id(values) == dest_id:
      return values
  raise Exception()

