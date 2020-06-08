
import os
import wave
with wave.open('/datasets/thchs_wav/wav/train/A11/A11_1.WAV', "rb") as wave_file:
  frame_rate = wave_file.getframerate()
  print(frame_rate)
        
x = '/datasets/LJSpeech-1.1-lite/wavs/LJ001-0001.wav'

with wave.open(x, "rb") as wave_file:
  frame_rate = wave_file.getframerate()
  print(frame_rate)

import argparse
import os
from parser.LJSpeechDatasetParser import LJSpeechDatasetParser

import epitran
import pandas as pd
from tqdm import tqdm

from ipa2symb import extract_from_sentence
from paths import preprocessed_file, preprocessed_file_debug, symbols_path, symbols_path_info
from text.adjustments import normalize_text
from text.conversion.SymbolConverter import get_from_file

csv_separator = '\t'


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('-b', '--base_dir', type=str, help='base directory', default='/datasets/models/taco2pt_ipa')
  parser.add_argument('-d', '--ljspeech', type=str, help='LJSpeech dataset directory', default='/datasets/LJSpeech-1.1')
  parser.add_argument('-i', '--ipa', type=str, help='transcribe to IPA', default='true')

  args = parser.parse_args()

  s = get_from_file(os.path.join(args.base_dir, symbols_path))
  s.plot(os.path.join(args.base_dir, symbols_path_info))