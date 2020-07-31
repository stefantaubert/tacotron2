from src.paths import get_wavs_dir, wavs_file_name
from src.common.utils import load_csv, save_csv
import os

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

def to_values(i, name, speaker_name, text, wav_path, duration):
  return (i, name, speaker_name, text, wav_path, duration)

def save_data(base_dir: str, name: str, data: list):
  dest_file_path = os.path.join(get_wavs_dir(base_dir, name), wavs_file_name)
  save_csv(data, dest_file_path)

def parse_data(base_dir: str, name: str):
  dest_file_path = os.path.join(get_wavs_dir(base_dir, name), wavs_file_name)
  return load_csv(dest_file_path).values

def already_exists(base_dir: str, name: str):
  dest_dir = get_wavs_dir(base_dir, name, create=False)
  exists = os.path.exists(dest_dir)
  return exists
