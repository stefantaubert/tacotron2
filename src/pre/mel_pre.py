"""
input: wav data
output: mel data
"""
import os

from tqdm import tqdm
from argparse import ArgumentParser
import torch
from src.common.utils import load_csv, save_csv
from src.paths import get_mels_dir, mels_file_name
from src.pre.mel_parser import MelParser
from src.tacotron.hparams import create_hparams
from src.pre.wav_pre_io import parse_data, get_wav, get_basename, get_id, get_speaker_name, get_text, get_duration
from src.pre.mel_pre_io import to_values, already_exists, save_data

def init_calc_mels_parser(parser: ArgumentParser):
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--origin_name', type=str, required=True)
  parser.add_argument('--destination_name', type=str, required=True)
  parser.add_argument('--custom_hparams', type=str)
  return __calc_mels

def __calc_mels(base_dir: str, origin_name: str, destination_name: str, custom_hparams: str):
  if not already_exists(base_dir, destination_name):
    result = []
    haparms = create_hparams(custom_hparams)
    mel_parser = MelParser(haparms)

    data = parse_data(base_dir, origin_name)
    dest_dir = get_mels_dir(base_dir, destination_name)
    # with torch.no_grad():
    print("Calculating mels...")
    for values in tqdm(data):
      mel_tensor = mel_parser.get_mel_tensor_from_file(get_wav(values))
      mel_path = os.path.join(dest_dir, "{}_{}.pt".format(get_id(values), get_basename(values)))
      torch.save(mel_tensor, mel_path)

      result.append(to_values(
        i=get_id(values),
        name=get_basename(values),
        speaker_name=get_speaker_name(values),
        text=get_text(values),
        wav_path=get_wav(values),
        mel_path=mel_path,
        duration=get_duration(values)
      ))

    save_data(base_dir, destination_name, result)
    print("Dataset saved.")

if __name__ == "__main__":
  __calc_mels(
    base_dir="/datasets/models/taco2pt_v2",
    origin_name="thchs_22050kHz_normalized_nosil",
    destination_name="thchs",
    custom_hparams=None,
  )

  # __calc_mels(
  #   base_dir="/datasets/models/taco2pt_v2",
  #   origin_name="ljs_22050kHz",
  #   destination_name="ljs",
  # )

  # __calc_mels(
  #   base_dir="/datasets/models/taco2pt_v2",
  #   origin_name="thchs_kaldi_22050kHz_nosil",
  #   destination_name="thchs_kaldi"
  # )
