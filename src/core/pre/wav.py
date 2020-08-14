"""
calculate wav duration and sampling rate
"""

import os
from dataclasses import dataclass
from typing import List

from scipy.io.wavfile import read
from tqdm import tqdm

from src.core.common import (get_duration_s, get_duration_s_file,
                                    normalize_file, remove_silence_file,
                                    upsample_file)
from src.core.common import load_csv, save_csv
from src.core.pre import DsData, DsDataList
from src.core.common import get_chunk_name


@dataclass()
class WavData:
  entry_id: int
  wav: str
  duration: float
  sr: int

  def __repr__(self):
    return str(self.entry_id)
  
class WavDataList(List[WavData]):
  def save(self, file_path: str):
    save_csv(self, file_path)

  @classmethod
  def load(cls, file_path: str):
    data = load_csv(file_path, WavData)
    return cls(data)

def preprocess(data: DsDataList) -> WavDataList:
  result = WavDataList()
  
  values: DsData
  for values in tqdm(data):
    sampling_rate, wav = read(values.wav_path)
    duration = get_duration_s(wav, sampling_rate)
    result.append(WavData(values.entry_id, values.wav_path, duration, sampling_rate))

  return result

def upsample(data: WavDataList, dest_dir: str, new_rate: int) -> WavDataList:
  assert os.path.isdir(dest_dir)
  result = WavDataList()

  values: WavData
  for values in tqdm(data):
    chunk_dir = os.path.join(dest_dir, get_chunk_name(values.entry_id, chunksize=500, maximum=len(data) - 1))
    os.makedirs(chunk_dir, exist_ok=True)
    dest_wav_path = os.path.join(chunk_dir, f"{values!r}.wav")
    # todo assert not is_overamp
    upsample_file(values.wav, dest_wav_path, new_rate)
    result.append(WavData(values.entry_id, dest_wav_path, values.duration, new_rate))

  return result

def remove_silence(data: WavDataList, dest_dir: str, chunk_size: int, threshold_start: float, threshold_end: float, buffer_start_ms: float, buffer_end_ms: float) -> WavDataList:
  assert os.path.isdir(dest_dir)
  result = WavDataList()

  values: WavData
  for values in tqdm(data):
    chunk_dir = os.path.join(dest_dir, get_chunk_name(values.entry_id, chunksize=500, maximum=len(data) - 1))
    os.makedirs(chunk_dir, exist_ok=True)
    dest_wav_path = os.path.join(chunk_dir, f"{values!r}.wav")
    new_duration = remove_silence_file(
      in_path = values.wav,
      out_path = dest_wav_path,
      chunk_size = chunk_size,
      threshold_start = threshold_start,
      threshold_end = threshold_end,
      buffer_start_ms = buffer_start_ms,
      buffer_end_ms = buffer_end_ms
    )
    result.append(WavData(values.entry_id, dest_wav_path, new_duration, values.sr))

  return result

def normalize(data: WavDataList, dest_dir: str) -> WavDataList:
  assert os.path.isdir(dest_dir)
  result = WavDataList()
  
  values: WavData
  for values in tqdm(data):
    chunk_dir = os.path.join(dest_dir, get_chunk_name(values.entry_id, chunksize=500, maximum=len(data) - 1))
    os.makedirs(chunk_dir, exist_ok=True)
    dest_wav_path = os.path.join(chunk_dir, f"{values!r}.wav")
    normalize_file(values.wav, dest_wav_path)
    result.append(WavData(values.entry_id, dest_wav_path, values.duration, values.sr))

  return result
