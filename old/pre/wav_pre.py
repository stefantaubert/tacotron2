"""
unify datasources and calculate wav duration
"""

import os

from tqdm import tqdm

from src.core.common.audio.utils import get_duration_s_file
from argparse import ArgumentParser
from src.pre.wav_pre_io import save_data, already_exists, WavData, WavDataList
from src.pre.parser.pre_data import PreDataList, PreData

def __read_wavs(base_dir: str, ds_name: str, data: PreDataList):
  result: WavDataList = []
  print("Reading durations...")
  values: PreData
  for i, values in enumerate(tqdm(data)):
    duration = get_duration_s_file(values.wav_path)
    wav_data = WavData(i, values.name, values.speaker_name, values.text, values.wav_path, duration)
    result.append(wav_data)
  
  save_data(base_dir, ds_name, result)
  print("Dataset saved.")

def __read_wavs_ds(base_dir: str, name: str, path: str, parse):
  if not already_exists(base_dir, name):
    data = parse(path)
    __read_wavs(base_dir, name, data)
  else:
    print("Nothing to do.")

def init_thchs_parser(parser: ArgumentParser):
  parser.add_argument('--path', type=str, required=True, help='THCHS dataset directory')
  parser.add_argument('--auto_dl', action="store_true")
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--name', type=str, required=True, default='thchs_16000kHz')
  return __read_wavs_thchs

def __read_wavs_thchs(base_dir: str, name: str, path: str, auto_dl: bool):
  from src.pre.parser.thchs import parse, ensure_downloaded
  if auto_dl: ensure_downloaded(path)
  __read_wavs_ds(base_dir, name, path, parse)
  
def init_thchs_kaldi_parser(parser: ArgumentParser):
  parser.add_argument('--path', type=str, required=True, help='THCHS dataset directory')
  parser.add_argument('--auto_dl', action="store_true")
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--name', type=str, required=True, default='thchs_kaldi_16000kHz')
  return __read_wavs_thchs_kaldi

def __read_wavs_thchs_kaldi(base_dir: str, name: str, path: str, auto_dl: bool):
  from src.pre.parser.thchs_kaldi import parse, ensure_downloaded
  if auto_dl: ensure_downloaded(path)
  __read_wavs_ds(base_dir, name, path, parse)

def init_ljs_parser(parser: ArgumentParser):
  parser.add_argument('--path', type=str, required=True, help='LJS dataset directory')
  parser.add_argument('--auto_dl', action="store_true")
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--name', type=str, required=True, default='ljs_22050kHz')
  return __read_wavs_ljs

def __read_wavs_ljs(base_dir: str, name: str, path: str, auto_dl: bool):
  from src.pre.parser.ljs import parse, ensure_downloaded
  if auto_dl: ensure_downloaded(path)
  __read_wavs_ds(base_dir, name, path, parse)

if __name__ == "__main__":
 
  __read_wavs_thchs(
    base_dir="/datasets/models/taco2pt_v2",
    path="/datasets/thchs_wav",
    name="thchs_16000kHz-test",
    auto_dl=True,
  )

  # __read_wavs_ljs(
  #   base_dir="/datasets/models/taco2pt_v2",
  #   path="/datasets/LJSpeech-1.1",
  #   name="ljs_22050kHz",
  #   auto_dl=True,
  # )

  # __read_wavs_thchs_kaldi(
  #   base_dir="/datasets/models/taco2pt_v2",
  #   path="/datasets/THCHS-30",
  #   name="thchs_kaldi_16000kHz",
  #   auto_dl=True,
  # )
