from src.paths import get_wavs_dir, wavs_file_name
import os
from dataclasses import dataclass
from typing import List
from src.common.utils import load_csv, save_csv


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
  dest_file_path = os.path.join(get_wavs_dir(base_dir, name), wavs_file_name)
  save_csv(data, dest_file_path)

def parse_data(base_dir: str, name: str) -> WavDataList:
  dest_file_path = os.path.join(get_wavs_dir(base_dir, name), wavs_file_name)
  return load_csv(dest_file_path, WavData)

def already_exists(base_dir: str, name: str):
  dest_dir = get_wavs_dir(base_dir, name, create=False)
  exists = os.path.exists(dest_dir)
  return exists
