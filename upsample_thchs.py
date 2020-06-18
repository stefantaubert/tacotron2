import argparse
from parser.thchs_parser import parse as parse_thchs
from shutil import copyfile
from tqdm import tqdm
import librosa    
from scipy.io import wavfile
import scipy.signal as sps
import os
from pathlib import Path

def create_parent_folder(file: str):
  path = Path(file)
  os.makedirs(path.parent, exist_ok=True)

def convert(origin, dest):
  parsed_data = parse_thchs(origin)

  a = os.path.join(origin, 'doc/trans/train.word.txt')
  b = os.path.join(dest, 'doc/trans/train.word.txt')
  create_parent_folder(b)
  copyfile(a, b)

  a = os.path.join(origin, 'doc/trans/test.word.txt')
  b = os.path.join(dest, 'doc/trans/test.word.txt')
  copyfile(a, b)

  new_rate = 22050

  for _, speaker_name, basename, wav_path, chn in tqdm(parsed_data):
    dest_wav_path = wav_path.replace(origin, dest)
    create_parent_folder(dest_wav_path)
    new_data, _ = librosa.load(wav_path, sr=new_rate)
    wavfile.write(dest_wav_path, new_rate, new_data)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_src_dir', type=str, help='THCHS dataset directory')
  parser.add_argument('--data_dest_dir', type=str, help='THCHS destination directory')
  parser.add_argument('--debug', type=str, default="true")

  args = parser.parse_args()

  debug = str.lower(args.debug) == 'true'

  if debug:
    args.data_src_dir = '/datasets/thchs_wav'
    args.data_dest_dir = '/datasets/thchs_wav_22050'
  
  convert(args.data_src_dir, args.data_dest_dir)
