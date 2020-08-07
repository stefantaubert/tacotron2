from src.tacotron.prepare_ds_ms_io import (filelist_file_log_name, filelist_file_name,
                       filelist_speakers_name, filelist_symbols_file_name,
                       filelist_test_file_name, filelist_training_file_name,
                       filelist_validation_file_name,
                       filelist_weights_file_name, get_filelist_dir)
import os
from src.common.utils import load_csv, save_csv
from src.common.train_log import log
import numpy as np
from dataclasses import dataclass
from typing import List
import random
@dataclass()
class PreparedData:
  i: int
  basename: str
  wav_path: str
  duration: float

PreparedDataList = List[PreparedData]

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

def parse_trainset(training_dir_path: str) -> PreparedDataList:
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

def get_random_val_utterance(training_dir_path: str) -> PreparedData:
  dataset = parse_testset(training_dir_path)
  return __get_random_utterance(dataset)

def get_random_test_utterance(training_dir_path: str) -> PreparedData:
  dataset = parse_testset(training_dir_path)
  return __get_random_utterance(dataset)

def __get_random_utterance(dataset: PreparedDataList) -> PreparedData:
  random_value: PreparedData = random.choice(dataset)
  return random_value

def get_values_entry(training_dir_path, dest_id: int) -> PreparedData:
  wholeset = parse_wholeset(training_dir_path)
  values: PreparedData
  for values in wholeset:
    if values.i == dest_id:
      return values
  raise Exception()

