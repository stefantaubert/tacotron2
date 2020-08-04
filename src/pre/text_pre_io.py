import os

from src.common.utils import load_csv, save_csv, save_json
from collections import OrderedDict
from src.paths import (ds_preprocessed_file_name, ds_preprocessed_symbols_name,
                       get_all_speakers_path, get_all_symbols_path, get_ds_dir,
                       get_mels_dir, mels_file_name)
from src.text.symbol_converter import load_from_file, SymbolConverter

def get_basename(values: tuple):
  wav = values[0]
  return wav

def get_mel_path(values: tuple):
  val = values[1]
  return val

def get_serialized_symbol_ids(values: tuple):
  val = values[2]
  return val

def get_duration(values: tuple):
  val = values[3]
  return val

def get_text(values: tuple):
  val = values[4]
  return val

def get_ipa(values: tuple):
  val = values[5]
  return val

def get_symbols_str(values: tuple):
  val = values[6]
  return val

def to_values(basename, mel_path, serialized_symbol_ids, duration, text, ipa, symbols_str):
  return (basename, mel_path, serialized_symbol_ids, duration, text, ipa, symbols_str)

def save_symbols(base_dir: str, ds_name: str, speaker: str, conv: SymbolConverter):
  ds_dir = get_ds_dir(base_dir, ds_name, speaker, create=True)
  ds_symbols_path = os.path.join(ds_dir, ds_preprocessed_symbols_name)
  conv.dump(ds_symbols_path)

def parse_symbols(base_dir: str, ds_name: str, speaker: str) -> SymbolConverter:
  speaker_dir_path = get_ds_dir(base_dir, ds_name, speaker)
  symbols_path = os.path.join(speaker_dir_path, ds_preprocessed_symbols_name)
  conv = load_from_file(symbols_path)
  return conv

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

def parse_data(base_dir: str, ds_name: str, speaker: str):
  speaker_dir_path = get_ds_dir(base_dir, ds_name, speaker)
  prepr_path = os.path.join(speaker_dir_path, ds_preprocessed_file_name)
  speaker_data = load_csv(prepr_path)
  return speaker_data.values

def already_exists(base_dir: str, ds_name: str):
  all_symbols_path = get_all_symbols_path(base_dir, ds_name)
  all_speakers_path = get_all_speakers_path(base_dir, ds_name)
  
  already_preprocessed = os.path.exists(all_speakers_path) and os.path.exists(all_symbols_path)
  return already_preprocessed
