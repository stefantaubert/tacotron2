import os
from src.common.utils import get_subdir

pre_dir = 'pre'
pre_data_file = 'data.csv'
pre_speakers_file = 'speakers.json'
pre_speakers_log_file = 'speakers_log.json'

wav_dir = "wav"
wav_data_file = "data.csv"

mel_dir = "mel"
mel_data_file = "data.csv"

text_dir = "text"
text_data_file = "data.csv"
text_symbols_json = "symbols.json"
text_symbol_converter = "symbol_ids.json"

def get_ds_dir(base_dir: str, ds_name: str, create: bool = False):
  return get_subdir(base_dir, os.path.join(pre_dir, ds_name), create)

def get_ds_csv(base_dir: str, ds_name: str):
  return os.path.join(get_ds_dir(base_dir, ds_name, True), pre_data_file)

def get_speakers_json(base_dir: str, ds_name: str):
  return os.path.join(get_ds_dir(base_dir, ds_name, True), pre_speakers_file)

def get_speakers_log_json(base_dir: str, ds_name: str):
  return os.path.join(get_ds_dir(base_dir, ds_name, True), pre_speakers_log_file)

def get_wav_subdir(base_dir: str, ds_name: str, sub_name: str, create: bool = False):
  return get_subdir(base_dir, os.path.join(pre_dir, ds_name, wav_dir, sub_name), create)

def get_wav_csv(base_dir: str, ds_name: str, sub_name: str):
  subdir = get_wav_subdir(base_dir, ds_name, sub_name, create=True)
  return os.path.join(subdir, wav_data_file)

def get_mel_subdir(base_dir: str, ds_name: str, sub_name: str, create: bool = False):
  return get_subdir(base_dir, os.path.join(pre_dir, ds_name, mel_dir, sub_name), create)

def get_mel_csv(base_dir: str, ds_name: str, sub_name: str):
  subdir = get_mel_subdir(base_dir, ds_name, sub_name, create=True)
  return os.path.join(subdir, mel_data_file)

def get_text_subdir(base_dir: str, ds_name: str, sub_name: str, create: bool = False):
  return get_subdir(base_dir, os.path.join(pre_dir, ds_name, text_dir, sub_name), create)

def get_text_csv(base_dir: str, ds_name: str, sub_name: str):
  subdir = get_text_subdir(base_dir, ds_name, sub_name, create=True)
  return os.path.join(subdir, text_data_file)

def get_text_symbols_json(base_dir: str, ds_name: str, sub_name: str):
  subdir = get_text_subdir(base_dir, ds_name, sub_name, create=True)
  return os.path.join(subdir, text_symbols_json)

def get_text_symbol_converter(base_dir: str, ds_name: str, sub_name: str):
  subdir = get_text_subdir(base_dir, ds_name, sub_name, create=True)
  return os.path.join(subdir, text_symbol_converter)
