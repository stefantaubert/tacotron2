import os

from tqdm import tqdm

import torch
from src.common.utils import load_csv, save_csv
from src.paths import get_mels_dir, mels_file_name
from src.pre.mel_parser import MelParser
from src.pre.wav_pre import parse_data as parse_data_wav


def save_data(base_dir: str, name: str, data: list):
  dest_file_path = os.path.join(get_mels_dir(base_dir, name), mels_file_name)
  save_csv(data, dest_file_path)

def parse_data(base_dir: str, name: str):
  dest_file_path = os.path.join(get_mels_dir(base_dir, name), mels_file_name)
  return load_csv(dest_file_path).values

def init_calc_mels_parser(parser):
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--origin_name', type=str, required=True)
  parser.add_argument('--destination_name', type=str, required=True)
  parser.add_argument('--custom_hparams', type=str)
  return __calc_mels

def __calc_mels(base_dir: str, origin_name: str, destination_name: str, custom_hparams: str):
  result = []
  mel_parser = MelParser(custom_hparams)
  data = parse_data_wav(base_dir, origin_name)
  dest_dir = get_mels_dir(base_dir, destination_name)
  # with torch.no_grad():
  print("Calculating mels...")
  for i, values in tqdm(enumerate(data), total=len(data)):
    name, speaker_name, text, wav_path = values[0], values[1], values[2], values[3]
    mel_path = os.path.join(dest_dir, "{}.pt".format(i))
    mel, _, duration = mel_parser.get_mel(wav_path)
    torch.save(mel, mel_path)
    result.append((name, speaker_name, text, mel_path, duration))
  save_data(base_dir, destination_name, result)
  print("Dataset saved.")

if __name__ == "__main__":
  __calc_mels(
    base_dir="/datasets/models/taco2pt_v2",
    origin_name="ljs",
    destination_name="ljs",
  )

  __calc_mels(
    base_dir="/datasets/models/taco2pt_v2",
    origin_name="thchs",
    destination_name="thchs",
  )

  __calc_mels(
    base_dir="/datasets/models/taco2pt_v2",
    origin_name="thchs_kaldi",
    destination_name="thchs_kaldi"
  )
