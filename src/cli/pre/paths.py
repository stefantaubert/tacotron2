import os
from src.common.utils import get_subdir

_pre_dir = 'pre'
_pre_data_file = 'data.csv'
_pre_speakers_file = 'speakers.json'
_pre_speakers_log_file = 'speakers_log.json'

_wav_dir = "wav"
_wav_data_file = "data.csv"

_mel_dir = "mel"
_mel_data_file = "data.csv"

_text_dir = "text"
_text_data_file = "data.csv"
_text_symbols_json = "symbols.json"
_text_symbol_converter = "symbol_ids.json"

def get_ds_dir(base_dir: str, ds_name: str, create: bool = False):
  return get_subdir(base_dir, os.path.join(_pre_dir, ds_name), create)

def get_ds_csv(base_dir: str, ds_name: str):
  return os.path.join(get_ds_dir(base_dir, ds_name, True), _pre_data_file)

def get_speakers_json(base_dir: str, ds_name: str):
  return os.path.join(get_ds_dir(base_dir, ds_name, True), _pre_speakers_file)

def get_speakers_log_json(base_dir: str, ds_name: str):
  return os.path.join(get_ds_dir(base_dir, ds_name, True), _pre_speakers_log_file)

def get_wav_subdir(base_dir: str, ds_name: str, sub_name: str, create: bool = False):
  return get_subdir(base_dir, os.path.join(_pre_dir, ds_name, _wav_dir, sub_name), create)

def get_wav_csv(base_dir: str, ds_name: str, sub_name: str):
  subdir = get_wav_subdir(base_dir, ds_name, sub_name, create=True)
  return os.path.join(subdir, _wav_data_file)

def get_mel_subdir(base_dir: str, ds_name: str, sub_name: str, create: bool = False):
  return get_subdir(base_dir, os.path.join(_pre_dir, ds_name, _mel_dir, sub_name), create)

def get_mel_csv(base_dir: str, ds_name: str, sub_name: str):
  subdir = get_mel_subdir(base_dir, ds_name, sub_name, create=True)
  return os.path.join(subdir, _mel_data_file)

def get_text_subdir(base_dir: str, ds_name: str, sub_name: str, create: bool = False):
  return get_subdir(base_dir, os.path.join(_pre_dir, ds_name, _text_dir, sub_name), create)

def get_text_csv(base_dir: str, ds_name: str, sub_name: str):
  subdir = get_text_subdir(base_dir, ds_name, sub_name, create=True)
  return os.path.join(subdir, _text_data_file)

def get_text_symbols_json(base_dir: str, ds_name: str, sub_name: str):
  subdir = get_text_subdir(base_dir, ds_name, sub_name, create=True)
  return os.path.join(subdir, _text_symbols_json)

def get_text_symbol_converter(base_dir: str, ds_name: str, sub_name: str):
  subdir = get_text_subdir(base_dir, ds_name, sub_name, create=True)
  return os.path.join(subdir, _text_symbol_converter)
