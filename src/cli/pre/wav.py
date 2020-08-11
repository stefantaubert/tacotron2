import os
from argparse import ArgumentParser
from src.core.pre.ds import DsDataList
from src.core.pre.wav import WavDataList
from src.core.pre.wav import normalize as normalize_core
from src.core.pre.wav import preprocess as preprocess_core
from src.core.pre.wav import upsample as upsample_core
from src.core.pre.wav import remove_silence as remove_silence_core
from src.cli.pre.paths import get_wav_subdir, get_wav_csv
from src.cli.pre.ds import load_ds_csv

#region IO

def load_wav_csv(base_dir: str, ds_name: str, sub_name: str) -> WavDataList:
  origin_wav_data_path = get_wav_csv(base_dir, ds_name, sub_name)
  return WavDataList.load(origin_wav_data_path)
  
def save_wav_csv(base_dir: str, ds_name: str, sub_name: str, wav_data: WavDataList):
  wav_data_path = get_wav_csv(base_dir, ds_name, sub_name)
  wav_data.save(wav_data_path)

def wav_subdir_exists(base_dir: str, ds_name: str, sub_name: str):
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
  if wav_subdir_exists(base_dir, ds_name, sub_name):
    print("Already exists.")
  else:
    data = load_ds_csv(base_dir, ds_name)
    #wav_data_dir = get_pre_ds_wav_subname_dir(base_dir, ds_name, sub_name, create=False)
    wav_data = preprocess_core(data)
    save_wav_csv(base_dir, ds_name, sub_name, wav_data)

#endregion

#region Normalizing

def init_normalize_parser(parser: ArgumentParser):
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--ds_name', type=str, required=True)
  parser.add_argument('--origin_sub_name', type=str, required=True)
  parser.add_argument('--destination_sub_name', type=str, required=True)
  return normalize

def normalize(base_dir: str, ds_name: str, origin_sub_name: str, destination_sub_name: str):
  if wav_subdir_exists(base_dir, ds_name, destination_sub_name):
    print("Already exists.")
  else:
    data = load_wav_csv(base_dir, ds_name, origin_sub_name)
    wav_data_dir = get_wav_subdir(base_dir, ds_name, destination_sub_name, create=True)
    wav_data = normalize_core(data, wav_data_dir)
    save_wav_csv(base_dir, ds_name, destination_sub_name, wav_data)

#endregion

#region Upsampling

def init_upsample_parser(parser: ArgumentParser):
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--ds_name', type=str, required=True)
  parser.add_argument('--origin_sub_name', type=str, required=True)
  parser.add_argument('--destination_sub_name', type=str, required=True)
  parser.add_argument('--rate', type=int, required=True)
  return upsample

def upsample(base_dir: str, ds_name: str, origin_sub_name: str, destination_sub_name: str, rate: int):
  if wav_subdir_exists(base_dir, ds_name, destination_sub_name):
    print("Already exists.")
  else:
    data = load_wav_csv(base_dir, ds_name, origin_sub_name)
    wav_data_dir = get_wav_subdir(base_dir, ds_name, destination_sub_name, create=True)
    wav_data = upsample_core(data, wav_data_dir, rate)
    save_wav_csv(base_dir, ds_name, destination_sub_name, wav_data)

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
  return remove_silence

def remove_silence(base_dir: str, ds_name: str, origin_sub_name: str, destination_sub_name: str, chunk_size: int, threshold_start: float, threshold_end: float, buffer_start_ms: float, buffer_end_ms: float):
  if wav_subdir_exists(base_dir, ds_name, destination_sub_name):
    print("Already exists.")
  else:
    data = load_wav_csv(base_dir, ds_name, origin_sub_name)
    wav_data_dir = get_wav_subdir(base_dir, ds_name, destination_sub_name, create=True)
    wav_data = remove_silence_core(data, wav_data_dir, chunk_size, threshold_start, threshold_end, buffer_start_ms, buffer_end_ms)
    save_wav_csv(base_dir, ds_name, destination_sub_name, wav_data)

#endregion

if __name__ == "__main__":
  preprocess(
    base_dir="/datasets/models/taco2pt_v2",
    ds_name="thchs",
    sub_name="16000kHz",
  )

  normalize(
    base_dir="/datasets/models/taco2pt_v2",
    ds_name="thchs",
    origin_sub_name="16000kHz",
    destination_sub_name="16000kHz_normalized",
  )

  upsample(
    base_dir="/datasets/models/taco2pt_v2",
    ds_name="thchs",
    origin_sub_name="16000kHz_normalized",
    destination_sub_name="22050kHz_normalized",
    rate=22050,
  )

  remove_silence(
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