import os
from argparse import ArgumentParser

from src.cli.pre.ds import load_ds_csv
from src.cli.pre.paths import get_wav_csv, get_wav_subdir
from src.core.pre import (DsDataList, WavDataList, wavs_normalize,
                          wavs_preprocess, wavs_remove_silence, wavs_upsample)

#region IO

def load_wav_csv(base_dir: str, ds_name: str, sub_name: str) -> WavDataList:
  origin_wav_data_path = get_wav_csv(base_dir, ds_name, sub_name)
  return WavDataList.load(origin_wav_data_path)
  
def _save_wav_csv(base_dir: str, ds_name: str, sub_name: str, wav_data: WavDataList):
  wav_data_path = get_wav_csv(base_dir, ds_name, sub_name)
  wav_data.save(wav_data_path)

def _wav_subdir_exists(base_dir: str, ds_name: str, sub_name: str):
  wav_data_dir = get_wav_subdir(base_dir, ds_name, sub_name, create=False)
  return os.path.exists(wav_data_dir)

#endregion

#region Processing wavs

def init_pre_parser(parser: ArgumentParser):
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--ds_name', type=str, required=True)
  parser.add_argument('--sub_name', type=str, required=True)
  return preprocess

def preprocess(base_dir: str, ds_name: str, sub_name: str):
  if _wav_subdir_exists(base_dir, ds_name, sub_name):
    print("Already exists.")
  else:
    data = load_ds_csv(base_dir, ds_name)
    #wav_data_dir = get_pre_ds_wav_subname_dir(base_dir, ds_name, sub_name, create=False)
    wav_data = wavs_preprocess(data)
    _save_wav_csv(base_dir, ds_name, sub_name, wav_data)

#endregion

#region Normalizing

def init_normalize_parser(parser: ArgumentParser):
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--ds_name', type=str, required=True)
  parser.add_argument('--origin_sub_name', type=str, required=True)
  parser.add_argument('--destination_sub_name', type=str, required=True)
  return _normalize

def _normalize(base_dir: str, ds_name: str, origin_sub_name: str, destination_sub_name: str):
  if _wav_subdir_exists(base_dir, ds_name, destination_sub_name):
    print("Already exists.")
  else:
    data = load_wav_csv(base_dir, ds_name, origin_sub_name)
    wav_data_dir = get_wav_subdir(base_dir, ds_name, destination_sub_name, create=True)
    wav_data = wavs_normalize(data, wav_data_dir)
    _save_wav_csv(base_dir, ds_name, destination_sub_name, wav_data)

#endregion

#region Upsampling

def init_upsample_parser(parser: ArgumentParser):
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--ds_name', type=str, required=True)
  parser.add_argument('--origin_sub_name', type=str, required=True)
  parser.add_argument('--destination_sub_name', type=str, required=True)
  parser.add_argument('--rate', type=int, required=True)
  return _upsample

def _upsample(base_dir: str, ds_name: str, origin_sub_name: str, destination_sub_name: str, rate: int):
  if _wav_subdir_exists(base_dir, ds_name, destination_sub_name):
    print("Already exists.")
  else:
    data = load_wav_csv(base_dir, ds_name, origin_sub_name)
    wav_data_dir = get_wav_subdir(base_dir, ds_name, destination_sub_name, create=True)
    wav_data = wavs_upsample(data, wav_data_dir, rate)
    _save_wav_csv(base_dir, ds_name, destination_sub_name, wav_data)

#endregion

#region Remove Silence

def init_remove_silence_parser(parser: ArgumentParser):
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--ds_name', type=str, required=True)
  parser.add_argument('--origin_sub_name', type=str, required=True)
  parser.add_argument('--destination_sub_name', type=str, required=True)
  parser.add_argument('--chunk_size', type=int, required=True)
  parser.add_argument('--threshold_start', type=float, required=True)
  parser.add_argument('--threshold_end', type=float, required=True)
  parser.add_argument('--buffer_start_ms', type=float, help="amount of factors of chunk_size at the beginning and the end should be reserved", required=True)
  parser.add_argument('--buffer_end_ms', type=float, help="amount of factors of chunk_size at the beginning and the end should be reserved", required=True)
  return _remove_silence

def _remove_silence(base_dir: str, ds_name: str, origin_sub_name: str, destination_sub_name: str, chunk_size: int, threshold_start: float, threshold_end: float, buffer_start_ms: float, buffer_end_ms: float):
  if _wav_subdir_exists(base_dir, ds_name, destination_sub_name):
    print("Already exists.")
  else:
    data = load_wav_csv(base_dir, ds_name, origin_sub_name)
    wav_data_dir = get_wav_subdir(base_dir, ds_name, destination_sub_name, create=True)
    wav_data = wavs_remove_silence(data, wav_data_dir, chunk_size, threshold_start, threshold_end, buffer_start_ms, buffer_end_ms)
    _save_wav_csv(base_dir, ds_name, destination_sub_name, wav_data)

#endregion

if __name__ == "__main__":
  preprocess(
    base_dir="/datasets/models/taco2pt_v2",
    ds_name="thchs",
    sub_name="16000kHz",
  )

  _normalize(
    base_dir="/datasets/models/taco2pt_v2",
    ds_name="thchs",
    origin_sub_name="16000kHz",
    destination_sub_name="16000kHz_normalized",
  )

  _upsample(
    base_dir="/datasets/models/taco2pt_v2",
    ds_name="thchs",
    origin_sub_name="16000kHz_normalized",
    destination_sub_name="22050kHz_normalized",
    rate=22050,
  )

  _remove_silence(
    base_dir="/datasets/models/taco2pt_v2",
    ds_name="thchs",
    origin_sub_name="22050kHz_normalized",
    destination_sub_name="22050kHz_normalized_nosil",
    threshold_start = -20,
    threshold_end = -30,
    chunk_size = 5,
    buffer_start_ms = 100,
    buffer_end_ms = 150
  )
