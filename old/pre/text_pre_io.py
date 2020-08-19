import os

from src.core.common import load_csv, save_csv, save_json, parse_json
from collections import OrderedDict
from src.core.common import get_subdir
from src.text.symbol_converter import load_from_file, SymbolConverter
from typing import List
from src.core.common import load_csv, save_csv
from dataclasses import dataclass

@dataclass()
class TextData:
  i: int
  basename: str
  wav_path: str
  mel_path: str
  serialized_symbol_ids: str
  duration: float
  text: str
  ipa: str
  symbols_str: str

TextDataList = List[TextData]

ds_dir = 'ds'
ds_preprocessed_file_name = 'filelist.csv'
ds_preprocessed_symbols_name = 'symbols.json'
ds_preprocessed_all_symbols_name = 'all_symbols.json'
ds_preprocessed_all_speakers_name = 'all_speakers.json'

def get_all_symbols_path(base_dir: str, name: str, create: bool = False) -> str:
  path = os.path.join(get_subdir(base_dir, os.path.join(ds_dir, name), create), ds_preprocessed_all_symbols_name)
  return path

def get_all_speakers_path(base_dir: str, name: str, create: bool = False) -> str:
  path = os.path.join(get_subdir(base_dir, os.path.join(ds_dir, name), create), ds_preprocessed_all_speakers_name)
  return path

def get_ds_dir(base_dir: str, name: str, speaker: str, create: bool = False) -> str:
  return get_subdir(base_dir, os.path.join(ds_dir, name, str(speaker)), create)

def save_symbols(base_dir: str, ds_name: str, speaker: str, conv: SymbolConverter):
  ds_dir = get_ds_dir(base_dir, ds_name, speaker, create=True)
  ds_symbols_path = os.path.join(ds_dir, ds_preprocessed_symbols_name)
  conv.dump(ds_symbols_path)

def parse_symbols(base_dir: str, ds_name: str, speaker: str) -> SymbolConverter:
  speaker_dir_path = get_ds_dir(base_dir, ds_name, speaker)
  symbols_path = os.path.join(speaker_dir_path, ds_preprocessed_symbols_name)
  conv = load_from_file(symbols_path)
  return conv

def expand_speakers(base_dir, ds_speakers):
  # expand all
  expanded_speakers = []
  for ds, speaker_name, _ in ds_speakers:
    if speaker_name == 'all':
      all_speakers_path = get_all_speakers_path(base_dir, ds)
      all_speakers = parse_json(all_speakers_path)
      all_speakers = sorted(all_speakers.keys())
      for speaker_name in all_speakers:
        expanded_speakers.append((ds, speaker_name))
    else:
      expanded_speakers.append((ds, speaker_name))

  expanded_speakers = list(sorted(set(expanded_speakers)))
  return expanded_speakers

def save_all_symbols(base_dir: str, ds_name: str, all_symbols: OrderedDict):
  all_symbols_path = get_all_symbols_path(base_dir, ds_name, create=True)
  save_json(all_symbols_path, all_symbols)

def save_all_speakers(base_dir: str, ds_name: str, all_speakers: OrderedDict):
  all_speakers_path = get_all_speakers_path(base_dir, ds_name, create=True)
  save_json(all_speakers_path, all_speakers)

def save_data(base_dir: str, ds_name: str, speaker: str, data: TextDataList):
  ds_dir = get_ds_dir(base_dir, ds_name, speaker, create=True)
  dest = os.path.join(ds_dir, ds_preprocessed_file_name)
  save_csv(data, dest)

def parse_data(base_dir: str, ds_name: str, speaker: str) -> TextDataList:
  speaker_dir_path = get_ds_dir(base_dir, ds_name, speaker)
  prepr_path = os.path.join(speaker_dir_path, ds_preprocessed_file_name)
  return load_csv(prepr_path, TextData)

def already_exists(base_dir: str, ds_name: str):
  all_symbols_path = get_all_symbols_path(base_dir, ds_name)
  all_speakers_path = get_all_speakers_path(base_dir, ds_name)
  already_preprocessed = os.path.exists(all_speakers_path) and os.path.exists(all_symbols_path)
  return already_preprocessed