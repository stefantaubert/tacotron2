"""
calculate wav duration and sampling rate
"""

from logging import getLogger
import os
from dataclasses import dataclass

from scipy.io.wavfile import read
from tqdm import tqdm

from src.core.common.audio import (get_duration_s, normalize_file,
                                   remove_silence_file, upsample_file)
from src.core.common.taco_stft import STFTHParams, TacotronSTFT
from src.core.common.utils import GenericList, get_chunk_name
from src.core.pre.ds import DsData, DsDataList


@dataclass()
class WavData:
  entry_id: int
  wav: str
  duration: float
  sr: int

  def __repr__(self):
    return str(self.entry_id)


class WavDataList(GenericList[WavData]):
  def get_entry(self, entry_id: int) -> WavData:
    for entry in self.items():
      if entry.entry_id == entry_id:
        return entry
    raise Exception(f"Entry {entry_id} not found.")


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
    chunk_dir = os.path.join(dest_dir, get_chunk_name(
      values.entry_id, chunksize=500, maximum=len(data) - 1))
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
    chunk_dir = os.path.join(dest_dir, get_chunk_name(
      values.entry_id, chunksize=500, maximum=len(data) - 1))
    os.makedirs(chunk_dir, exist_ok=True)
    dest_wav_path = os.path.join(chunk_dir, f"{values!r}.wav")
    new_duration = remove_silence_file(
      in_path=values.wav,
      out_path=dest_wav_path,
      chunk_size=chunk_size,
      threshold_start=threshold_start,
      threshold_end=threshold_end,
      buffer_start_ms=buffer_start_ms,
      buffer_end_ms=buffer_end_ms
    )
    result.append(WavData(values.entry_id, dest_wav_path, new_duration, values.sr))

  return result


def remove_silence_plot(wav_path: str, out_path: str, chunk_size: int, threshold_start: float, threshold_end: float, buffer_start_ms: float, buffer_end_ms: float):
  remove_silence_file(
    in_path=wav_path,
    out_path=out_path,
    chunk_size=chunk_size,
    threshold_start=threshold_start,
    threshold_end=threshold_end,
    buffer_start_ms=buffer_start_ms,
    buffer_end_ms=buffer_end_ms
  )

  sampling_rate, _ = read(wav_path)

  hparams = STFTHParams()
  hparams.sampling_rate = sampling_rate
  plotter = TacotronSTFT(hparams, logger=getLogger())

  mel_orig = plotter.get_mel_tensor_from_file(wav_path)
  mel_trimmed = plotter.get_mel_tensor_from_file(out_path)

  return mel_orig, mel_trimmed


def normalize(data: WavDataList, dest_dir: str) -> WavDataList:
  assert os.path.isdir(dest_dir)
  result = WavDataList()

  values: WavData
  for values in tqdm(data):
    chunk_dir = os.path.join(dest_dir, get_chunk_name(
      values.entry_id, chunksize=500, maximum=len(data) - 1))
    os.makedirs(chunk_dir, exist_ok=True)
    dest_wav_path = os.path.join(chunk_dir, f"{values!r}.wav")
    normalize_file(values.wav, dest_wav_path)
    result.append(WavData(values.entry_id, dest_wav_path, values.duration, values.sr))

  return result
