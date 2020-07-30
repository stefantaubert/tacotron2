"""
unify datasources and calculate wav duration
"""

import os

from tqdm import tqdm

from src.common.audio.utils import get_duration
from src.common.utils import load_csv, save_csv
from src.paths import get_wavs_dir, wavs_file_name


def save_data(base_dir: str, name: str, data: list):
  dest_file_path = os.path.join(get_wavs_dir(base_dir, name), wavs_file_name)
  save_csv(data, dest_file_path)

def parse_data(base_dir: str, name: str):
  dest_file_path = os.path.join(get_wavs_dir(base_dir, name), wavs_file_name)
  return load_csv(dest_file_path).values

def read_wavs(base_dir: str, ds_name: str, data: list):
  result = []
  print("Reading durations...")
  for i, values in tqdm(enumerate(data), total=len(data)):
    assert len(values) == 4
    name, speaker_name, text, wav_path = values
    duration = get_duration(wav_path)
    result.append((name, speaker_name, text, wav_path, duration))
    
  save_data(base_dir, ds_name, data)
  print("Dataset saved.")

def read_wavs_thchs(base_dir: str, name: str, path: str):
  from src.pre.parser.thchs import parse
  data = parse(path)
  read_wavs(base_dir, name, data)

def read_wavs_thchs_kaldi(base_dir: str, name: str, path: str):
  from src.pre.parser.thchs_kaldi import parse
  data = parse(path)
  read_wavs(base_dir, name, data)

def read_wavs_ljs(base_dir: str, name: str, path: str):
  from src.pre.parser.ljs import parse
  data = parse(path)
  read_wavs(base_dir, name, data)

if __name__ == "__main__":
  read_wavs_ljs(
    base_dir="/datasets/models/taco2pt_v2/",
    path="/datasets/LJSpeech-1.1",
    name="ljs",
  )

  # read_wavs_thchs(
  #   base_dir="/datasets/models/taco2pt_v2/",
  #   path="/datasets/thchs_16bit_22050kHz_nosil",
  #   name="thchs",
  # )

  # read_wavs_thchs_kaldi(
  #   base_dir="/datasets/models/taco2pt_v2/",
  #   path="/datasets/thchs_16bit_22050kHz_nosil",
  #   name="thchs",
  # )
