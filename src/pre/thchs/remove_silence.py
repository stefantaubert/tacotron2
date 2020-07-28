import os
from pathlib import Path
from shutil import copyfile

import librosa
import numpy as np
import scipy.signal as sps
from scipy.io import wavfile
from tqdm import tqdm

from src.parser.thchs_kaldi_parser import parse as kaldi_parse
from src.parser.thchs_parser import parse as parse
from src.common.utils import create_parent_folder
from src.common.audio.remove_silence import remove_silence

def remove_silence_thchs(
    origin: str,
    dest: str,
    kaldi_version: bool,
    chunk_size: int,
    threshold_start: float,
    threshold_end: float,
    buffer_start_ms: float,
    buffer_end_ms: float
  ):

  if kaldi_version:
    parsed_data = kaldi_parse(origin)
  else:
    parsed_data = parse(origin)

  print("Removing silence at start and end of wav files...")
  for data in tqdm(parsed_data):
    wav_path = data[3]

    dest_wav_path = wav_path.replace(origin, dest)
    create_parent_folder(dest_wav_path)

    remove_silence(
      in_path = wav_path,
      out_path = dest_wav_path,
      chunk_size = chunk_size,
      threshold_start = threshold_start,
      threshold_end = threshold_end,
      buffer_start_ms = buffer_start_ms,
      buffer_end_ms = buffer_end_ms
    )

  if kaldi_version:
    for data in tqdm(parsed_data):
      sent_file = data[5]
      a = sent_file
      b = sent_file.replace(origin, dest)
      copyfile(a, b)
  else:
    a = os.path.join(origin, 'doc/trans/test.word.txt')
    b = os.path.join(dest, 'doc/trans/test.word.txt')
    create_parent_folder(b)
    copyfile(a, b)

    a = os.path.join(origin, 'doc/trans/train.word.txt')
    b = os.path.join(dest, 'doc/trans/train.word.txt')
    copyfile(a, b)

from src.parser.thchs_parser import parse, exists
from src.parser.thchs_kaldi_parser import parse as kaldi_parse, exists as kaldi_exists

def main(data_src_dir, data_dest_dir, kaldi_version, chunk_size, threshold_start, threshold_end, buffer_start_ms, buffer_end_ms):
  if kaldi_version:
    already_removed = kaldi_exists(data_dest_dir)
  else:
    already_removed = exists(data_dest_dir)
  
  if already_removed:
    print("Dataset is already without silence.")
  else:
    print("Saving to {}".format(data_dest_dir))
    
    remove_silence_thchs(
      origin=data_src_dir,
      dest=data_dest_dir,
      kaldi_version=kaldi_version,
      chunk_size = chunk_size,
      threshold_start = threshold_start,
      threshold_end = threshold_end,
      buffer_start_ms = buffer_start_ms,
      buffer_end_ms = buffer_end_ms
    )

    print("Finished.")

if __name__ == "__main__":
  main(
    data_src_dir='/datasets/thchs_16bit_22050kHz',
    data_dest_dir='/datasets/thchs_16bit_22050kHz_nosil',
    kaldi_version=False,
    chunk_size=5,
    threshold_start=-25,
    threshold_end=-35,
    buffer_start_ms=100,
    buffer_end_ms=150
  )