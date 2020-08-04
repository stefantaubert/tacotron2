import os
from collections import OrderedDict

import numpy as np

from src.common.train_log import log
from src.common.utils import (get_total_duration_min, load_csv, parse_json,
                              save_csv, save_json)
from src.paths import (ds_preprocessed_file_name, ds_preprocessed_symbols_name,
                       filelist_file_log_name, filelist_file_name,
                       filelist_speakers_name, filelist_symbols_file_name,
                       filelist_test_file_name, filelist_training_file_name,
                       filelist_validation_file_name,
                       filelist_weights_file_name, get_all_speakers_path,
                       get_ds_dir, get_filelist_dir, train_map_file)

__mel_path_idx = 1
__serialized_ids_idx = 2
__speaker_id_idx = 4

def get_speaker_id(values):
  return values[__speaker_id_idx]

def get_serialized_ids(values):
  return values[__serialized_ids_idx]

def get_mel_path(values):
  return values[__mel_path_idx]

def save_all_symbols(training_dir_path: str, conv):
  # symbols.json
  dest = os.path.join(get_filelist_dir(training_dir_path), filelist_symbols_file_name)
  conv.dump(dest)

def save_all_speakers(training_dir_path: str, speakers_info: OrderedDict):
  speakers_file = os.path.join(get_filelist_dir(training_dir_path), filelist_speakers_name)
  save_json(speakers_file, speakers_info)

def to_values(basename, mel_path, serialized_updated_ids, duration, speaker_id, speaker_name):
  return (basename, mel_path, serialized_updated_ids, duration, speaker_id, speaker_name)

def save_trainset(training_dir_path, dataset: list):
  dest_path = os.path.join(get_filelist_dir(training_dir_path), filelist_training_file_name)
  df = save_csv(dataset, dest_path)
  total_dur_min = get_total_duration_min(df, 3)
  log(training_dir_path, "{} => Size: {}, Duration: {:.2f}min / {:.2f}h".format(filelist_training_file_name, len(df), total_dur_min, total_dur_min / 60))

def save_testset(training_dir_path, dataset: list):
  dest_path = os.path.join(get_filelist_dir(training_dir_path), filelist_test_file_name)
  df = save_csv(dataset, dest_path)
  total_dur_min = get_total_duration_min(df, 3)
  log(training_dir_path, "{} => Size: {}, Duration: {:.2f}min / {:.2f}h".format(filelist_test_file_name, len(df), total_dur_min, total_dur_min / 60))

def save_validationset(training_dir_path, dataset: list):
  dest_path = os.path.join(get_filelist_dir(training_dir_path), filelist_validation_file_name)
  df = save_csv(dataset, dest_path)
  total_dur_min = get_total_duration_min(df, 3)
  log(training_dir_path, "{} => Size: {}, Duration: {:.2f}min / {:.2f}h".format(filelist_validation_file_name, len(df), total_dur_min, total_dur_min / 60))

def save_wholeset(training_dir_path, dataset: list):
  dest_path = os.path.join(get_filelist_dir(training_dir_path), filelist_file_name)
  df = save_csv(dataset, dest_path)
  total_dur_min = get_total_duration_min(df, 3)
  log(training_dir_path, "{} => Size: {}, Duration: {:.2f}min / {:.2f}h".format(filelist_file_name, len(df), total_dur_min, total_dur_min / 60))

def parse_traindata(training_dir_path: str):
  dest_path = os.path.join(get_filelist_dir(training_dir_path), filelist_training_file_name)
  return load_csv(dest_path).values

def parse_validationset(training_dir_path: str):
  dest_path = os.path.join(get_filelist_dir(training_dir_path), filelist_validation_file_name)
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
