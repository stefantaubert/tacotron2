import os
from dataclasses import dataclass
from typing import List
from src.core.common.utils import load_csv, save_csv
from src.core.common.utils import get_subdir

wavs_dir = 'wavs'
wavs_file_name = 'wavs.csv'

def get_wavs_dir(base_dir: str, name: str, create: bool = True) -> str:
  return get_subdir(base_dir, os.path.join(wavs_dir, name), create)

@dataclass()
class WavData:
  i: int
  basename: str
  speaker_name: str
  text: str
  wav: str
  duration: float

WavDataList = List[WavData]

def save_data(base_dir: str, name: str, data: WavDataList):
  dest_file_path = os.path.join(get_wavs_dir(base_dir, name, create=True), wavs_file_name)
  save_csv(data, dest_file_path)

def parse_data(base_dir: str, name: str) -> WavDataList:
  dest_file_path = os.path.join(get_wavs_dir(base_dir, name, create=False), wavs_file_name)
  return load_csv(dest_file_path, WavData)

def already_exists(base_dir: str, name: str):
  dest_dir = get_wavs_dir(base_dir, name, create=False)
  exists = os.path.exists(dest_dir)
  return exists
