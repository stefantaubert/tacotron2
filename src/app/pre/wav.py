import os
from functools import partial

from src.app.pre.ds import get_ds_dir, load_ds_csv
from src.core.common import get_subdir
from src.core.pre import (DsData, DsDataList, MelDataList, PreparedData,
                          PreparedDataList, SpeakersDict, SpeakersIdDict,
                          SpeakersLogDict, SymbolConverter, SymbolsDict,
                          TextDataList, WavDataList)
from src.core.pre import wavs_normalize as wavs_normalize_core
from src.core.pre import wavs_preprocess as wavs_preprocess_core
from src.core.pre import wavs_remove_silence as wavs_remove_silence_core
from src.core.pre import wavs_upsample as wavs_upsample_core

_wav_data_csv = "data.csv"

def _get_wav_root_dir(ds_dir: str, create: bool = False):
  return get_subdir(ds_dir, "wav", create)

def get_wav_dir(ds_dir: str, wav_name: str, create: bool = False):
  return get_subdir(_get_wav_root_dir(ds_dir, create), wav_name, create)

def load_wav_csv(wav_dir: str) -> WavDataList:
  path = os.path.join(wav_dir, _wav_data_csv)
  return WavDataList.load(path)
  
def save_wav_csv(wav_dir: str, wav_data: WavDataList):
  path = os.path.join(wav_dir, _wav_data_csv)
  wav_data.save(path)

def preprocess_wavs(base_dir: str, ds_name: str, wav_name: str):
  ds_dir = get_ds_dir(base_dir, ds_name)
  wav_dir = get_wav_dir(ds_dir, wav_name)
  if os.path.isdir(wav_dir):
    print("Already exists.")
  else:
    os.makedirs(wav_dir)
    data = load_ds_csv(ds_dir)
    wav_data = wavs_preprocess_core(data)
    save_wav_csv(wav_dir, wav_data)

def _wav_op(base_dir: str, ds_name: str, origin_wav_name: str, destination_wav_name: str, op):
  ds_dir = get_ds_dir(base_dir, ds_name)
  dest_wav_dir = get_wav_dir(ds_dir, destination_wav_name)
  if os.path.isdir(dest_wav_dir):
    print("Already exists.")
  else:
    orig_wav_dir = get_wav_dir(ds_dir, origin_wav_name)
    assert os.path.isdir(orig_wav_dir)
    data = load_wav_csv(orig_wav_dir)
    os.makedirs(dest_wav_dir)
    wav_data = op(data, dest_wav_dir)
    save_wav_csv(dest_wav_dir, wav_data)

def wavs_normalize(base_dir: str, ds_name: str, orig_wav_name: str, dest_wav_name: str):
  op = partial(wavs_normalize_core)
  _wav_op(base_dir, ds_name, orig_wav_name, dest_wav_name, op)
  
def wavs_upsample(base_dir: str, ds_name: str, orig_wav_name: str, dest_wav_name: str, rate: int):
  op = partial(wavs_upsample_core, new_rate=rate)
  _wav_op(base_dir, ds_name, orig_wav_name, dest_wav_name, op)

def wavs_remove_silence(base_dir: str, ds_name: str, orig_wav_name: str, dest_wav_name: str, chunk_size: int, threshold_start: float, threshold_end: float, buffer_start_ms: float, buffer_end_ms: float):
  op = partial(wavs_remove_silence_core, chunk_size=chunk_size, threshold_start=threshold_start, threshold_end=threshold_end, buffer_start_ms=buffer_start_ms, buffer_end_ms=buffer_end_ms)
  _wav_op(base_dir, ds_name, orig_wav_name, dest_wav_name, op)

if __name__ == "__main__":
  preprocess_wavs(
    base_dir="/datasets/models/taco2pt_v3",
    ds_name="ljs",
    wav_name="22050kHz",
  )

  preprocess_wavs(
    base_dir="/datasets/models/taco2pt_v3",
    ds_name="thchs",
    wav_name="16000kHz",
  )

  wavs_normalize(
    base_dir="/datasets/models/taco2pt_v3",
    ds_name="thchs",
    orig_wav_name="16000kHz",
    dest_wav_name="16000kHz_normalized",
  )

  wavs_remove_silence(
    base_dir="/datasets/models/taco2pt_v3",
    ds_name="thchs",
    orig_wav_name="16000kHz_normalized",
    dest_wav_name="16000kHz_normalized_nosil",
    threshold_start = -20,
    threshold_end = -30,
    chunk_size = 5,
    buffer_start_ms = 100,
    buffer_end_ms = 150
  )

  wavs_upsample(
    base_dir="/datasets/models/taco2pt_v3",
    ds_name="thchs",
    orig_wav_name="16000kHz_normalized_nosil",
    dest_wav_name="22050kHz_normalized_nosil",
    rate=22050,
  )
