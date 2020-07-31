import os

from src.common.utils import load_csv, save_csv, save_json
from collections import OrderedDict
from src.paths import (ds_preprocessed_file_name, ds_preprocessed_symbols_name,
                       get_all_speakers_path, get_all_symbols_path, get_ds_dir,
                       get_mels_dir, mels_file_name)

def get_id(values: tuple):
  wav = values[0]
  return wav

def get_basename(values: tuple):
  wav = values[1]
  return wav

def get_path(values: tuple):
  wav = values[4]
  return wav

def set_path(values: tuple, wav: str):
  values[4] = wav

def set_duration(values: tuple, duration: float):
  values[5] = duration

def to_values(basename, mel_path, serialized_symbol_ids, duration, text, ipa, symbols_str):
  return (basename, mel_path, serialized_symbol_ids, duration, text, ipa, symbols_str)

def save_symbols(base_dir: str, ds_name: str, speaker: str, conv):
  ds_dir = get_ds_dir(base_dir, ds_name, speaker, create=True)
  ds_symbols_path = os.path.join(ds_dir, ds_preprocessed_symbols_name)
  conv.dump(ds_symbols_path)

def parse_symbols(base_dir: str, name: str, speaker: str):
  pass

def save_all_symbols(base_dir: str, ds_name: str, all_symbols: OrderedDict):
  all_symbols_path = get_all_symbols_path(base_dir, ds_name)
  save_json(all_symbols_path, all_symbols)

def save_all_speakers(base_dir: str, ds_name: str, all_speakers: OrderedDict):
  all_speakers_path = get_all_speakers_path(base_dir, ds_name)
  save_json(all_speakers_path, all_speakers)

def save_data(base_dir: str, ds_name: str, speaker: str, data: list):
  ds_dir = get_ds_dir(base_dir, ds_name, speaker, create=True)
  dest = os.path.join(ds_dir, ds_preprocessed_file_name)
  save_csv(data, dest)

def parse_data(base_dir: str, name: str, speaker: str):
  pass

def already_exists(base_dir: str, ds_name: str):
  all_symbols_path = get_all_symbols_path(base_dir, ds_name)
  all_speakers_path = get_all_speakers_path(base_dir, ds_name)
  
  already_preprocessed = os.path.exists(all_speakers_path) and os.path.exists(all_symbols_path)
  return already_preprocessed
