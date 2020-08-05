import os

from src.common.utils import load_csv, save_csv, save_json
from collections import OrderedDict
from src.paths import (ds_preprocessed_file_name, ds_preprocessed_symbols_name,
                       get_all_speakers_path, get_all_symbols_path, get_ds_dir,
                       get_mels_dir, mels_file_name)
from src.text.symbol_converter import load_from_file, SymbolConverter

__id_idx = 0
__basename_idx = 1
__wav_path_idx = 2
__mel_path_idx = 3
__serialized_symbol_ids_idx = 4
__duration_idx = 5
__text_idx = 6
__ipa_idx = 7
__symbols_str_idx = 8

def get_id(values: tuple):
  return values[__id_idx]

def get_basename(values: tuple):
  return values[__basename_idx]

def get_wav_path(values: tuple):
  return values[__wav_path_idx]

def get_mel_path(values: tuple):
  return values[__mel_path_idx]

def get_serialized_symbol_ids(values: tuple):
  return values[__serialized_symbol_ids_idx]

def get_duration(values: tuple):
  return values[__duration_idx]

def get_text(values: tuple):
  return values[__text_idx]

def get_ipa(values: tuple):
  return values[__ipa_idx]

def get_symbols_str(values: tuple):
  return values[__symbols_str_idx]

def to_values(i, basename, wav_path, mel_path, serialized_symbol_ids, duration, text, ipa, symbols_str):
  return (i, basename, wav_path, mel_path, serialized_symbol_ids, duration, text, ipa, symbols_str)

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
