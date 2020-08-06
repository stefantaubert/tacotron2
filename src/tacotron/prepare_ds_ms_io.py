import random
import os
from collections import OrderedDict

import numpy as np

from src.common.utils import get_subdir
from src.common.train_log import log
from src.common.utils import (load_csv, parse_json,
                              save_csv, save_json)
from src.text.symbol_converter import load_from_file
from dataclasses import dataclass
from typing import List
import torch
from shutil import copyfile

@dataclass()
class PreparedData:
  i: int
  basename: str
  wav_path: str
  mel_path: str
  serialized_updated_ids: str
  duration: float
  speaker_id: int
  speaker_name: str

PreparedDataList = List[PreparedData]

filelist_dir = "filelist"
filelist_training_file_name = 'audio_text_train_filelist.csv'
filelist_test_file_name = 'audio_text_test_filelist.csv'
filelist_validation_file_name = 'audio_text_val_filelist.csv'
filelist_symbols_file_name = 'symbols.json'
filelist_file_name = 'filelist.csv'
filelist_file_log_name = 'filelist_log.csv'
filelist_speakers_name = 'speakers.json'
filelist_weights_file_name = 'weights.npy'

weights_map_file = 'weights_map.json'

def save_train_map(training_dir_path: str, map_path: str):
  assert map_path
  copyfile(map_path, os.path.join(training_dir_path, weights_map_file))

def parse_weights_map(training_dir_path: str):
  map_path = os.path.join(training_dir_path, weights_map_file)
  ipa_mapping = parse_json(map_path)
  return ipa_mapping

def remove_train_map(training_dir_path: str):
  map_file = os.path.join(training_dir_path, weights_map_file)
  if os.path.exists(map_file):
    os.remove(map_file)

def load_weights(training_dir_path: str):
  filelist_dir_path = get_filelist_dir(training_dir_path, create=False)
  weights_path = os.path.join(filelist_dir_path, filelist_weights_file_name)
  assert os.path.isfile(weights_path)
  log(training_dir_path, "Init weights from '{}'".format(weights_path))
  weights = np.load(weights_path)
  weights = torch.from_numpy(weights)
  return weights

def weights_map_exists(training_dir_path: str) -> bool:
  weights_path = os.path.join(get_filelist_dir(training_dir_path), filelist_weights_file_name)
  return os.path.exists(weights_path)

def get_symbols_path(training_dir_path: str) -> str:
  path = os.path.join(get_filelist_dir(training_dir_path), filelist_symbols_file_name)
  return path

def get_filelist_dir(training_dir_path: str, create: bool = True) -> str:
  return get_subdir(training_dir_path, filelist_dir, create)

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

def get_total_duration(values: PreparedDataList):
  x: PreparedData
  durations = [x.duration for x in values]
  duration = sum(durations)
  return duration
  
def save_trainset(training_dir_path, dataset: PreparedDataList):
  dest_path = os.path.join(get_filelist_dir(training_dir_path), filelist_training_file_name)
  save_csv(dataset, dest_path)
  total_dur_min = get_total_duration(dataset) / 60
  log(training_dir_path, "{} => Size: {}, Duration: {:.2f}min / {:.2f}h".format(filelist_training_file_name, len(dataset), total_dur_min, total_dur_min / 60))

def save_testset(training_dir_path, dataset: PreparedDataList):
  dest_path = os.path.join(get_filelist_dir(training_dir_path), filelist_test_file_name)
  save_csv(dataset, dest_path)
  total_dur_min = get_total_duration(dataset) / 60
  log(training_dir_path, "{} => Size: {}, Duration: {:.2f}min / {:.2f}h".format(filelist_test_file_name, len(dataset), total_dur_min, total_dur_min / 60))

def save_validationset(training_dir_path, dataset: PreparedDataList):
  dest_path = os.path.join(get_filelist_dir(training_dir_path), filelist_validation_file_name)
  save_csv(dataset, dest_path)
  total_dur_min = get_total_duration(dataset) / 60
  log(training_dir_path, "{} => Size: {}, Duration: {:.2f}min / {:.2f}h".format(filelist_validation_file_name, len(dataset), total_dur_min, total_dur_min / 60))

def save_wholeset(training_dir_path, dataset: PreparedDataList):
  dest_path = os.path.join(get_filelist_dir(training_dir_path), filelist_file_name)
  save_csv(dataset, dest_path)
  total_dur_min = get_total_duration(dataset) / 60
  log(training_dir_path, "{} => Size: {}, Duration: {:.2f}min / {:.2f}h".format(filelist_file_name, len(dataset), total_dur_min, total_dur_min / 60))

def parse_traindata(training_dir_path: str) -> PreparedDataList:
  dest_path = os.path.join(get_filelist_dir(training_dir_path), filelist_training_file_name)
  return load_csv(dest_path, PreparedData)

def parse_validationset(training_dir_path: str) -> PreparedDataList:
  dest_path = os.path.join(get_filelist_dir(training_dir_path), filelist_validation_file_name)
  return load_csv(dest_path, PreparedData)

def parse_testset(training_dir_path: str) -> PreparedDataList:
  dest_path = os.path.join(get_filelist_dir(training_dir_path), filelist_test_file_name)
  return load_csv(dest_path, PreparedData)

def parse_wholeset(training_dir_path: str) -> PreparedDataList:
  dest_path = os.path.join(get_filelist_dir(training_dir_path), filelist_file_name)
  return load_csv(dest_path, PreparedData)

def remove_weights_file(training_dir_path: str):
  weights_path = os.path.join(get_filelist_dir(training_dir_path), filelist_weights_file_name)
  if os.path.exists(weights_path):
    os.remove(weights_path)

def save_weights(training_dir_path: str, embedding):
  weights_path = os.path.join(get_filelist_dir(training_dir_path), filelist_weights_file_name)
  np.save(weights_path, embedding.weight.data.numpy())

def get_random_val_utterance(training_dir_path: str, custom_speaker: str) -> PreparedData:
  dataset = parse_testset(training_dir_path)
  return __get_random_utterance(dataset, custom_speaker)

def get_random_test_utterance(training_dir_path: str, custom_speaker: str) -> PreparedData:
  dataset = parse_testset(training_dir_path)
  return __get_random_utterance(dataset, custom_speaker)

def __get_random_utterance(dataset: PreparedDataList, custom_speaker: str) -> PreparedData:
  random_value: PreparedData = random.choice(dataset)
  if custom_speaker:
    while True:
      if random_value.speaker_name == custom_speaker:
        break
      random_value: PreparedData = random.choice(dataset)
  return random_value

def get_values_entry(training_dir_path, dest_id: int) -> PreparedData:
  wholeset = parse_wholeset(training_dir_path)
  values: PreparedData
  for values in wholeset:
    if values.i == dest_id:
      return values
  raise Exception()

