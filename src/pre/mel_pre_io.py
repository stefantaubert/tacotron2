from src.common.utils import get_subdir
from src.common.utils import load_csv, save_csv
import os

from dataclasses import dataclass
from typing import List

#region IO
mels_dir = 'mels'
mels_file_name = 'mels.csv'

def get_mels_dir(base_dir: str, name: str, create: bool = False) -> str:
  return get_subdir(base_dir, os.path.join(mels_dir, name), create)

@dataclass()
class MelData:
  i: int
  name: str
  speaker_name: str
  text: str
  wav_path: str
  mel_path: str
  duration: float

MelDataList = List[MelData]

def save_data(base_dir: str, name: str, data: MelDataList):
  dest_file_path = os.path.join(get_mels_dir(base_dir, name, create=True), mels_file_name)
  save_csv(data, dest_file_path)

def parse_data(base_dir: str, name: str) -> MelDataList:
  dest_file_path = os.path.join(get_mels_dir(base_dir, name), mels_file_name)
  return load_csv(dest_file_path, MelData)

def already_exists(base_dir: str, name: str):
  dest_dir = get_mels_dir(base_dir, name)
  exists = os.path.exists(dest_dir)
  return exists

#endregion