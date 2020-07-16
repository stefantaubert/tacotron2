import argparse
import os
from parser.thchs_parser import parse, exists, ensure_downloaded
from parser.thchs_kaldi_parser import parse as kaldi_parse, exists as kaldi_exists, ensure_downloaded as kaldi_ensure_downloaded

import epitran
import pandas as pd
from tqdm import tqdm
import librosa
import tarfile
import shutil
import tempfile
from pathlib import Path

from ipa2symb import extract_from_sentence
from paths import get_ds_dir, ds_preprocessed_file_name, ds_preprocessed_symbols_name, get_all_symbols_path, get_all_speakers_path
from text.adjustments import normalize_text
from text.symbol_converter import init_from_symbols, serialize_symbol_ids
from utils import csv_separator
import wget
from text.chn_tools import chn_to_ipa
from script_upsample_thchs import convert
from collections import Counter, OrderedDict
from utils import save_json, download_tar

def ensure_is_22050kHz(dir_path: str, data_conversion_dir: str, kaldi_version: bool):
  if kaldi_version:
    is_converted = kaldi_exists(data_conversion_dir)
  else:
    is_converted = exists(data_conversion_dir)

  if not is_converted:
    convert(dir_path, data_conversion_dir)

def preprocess(base_dir: str, data_dir: str, ds_name: str, ignore_tones: bool, ignore_arcs: bool, kaldi_version: bool):
  if kaldi_version:
    parsed_data = kaldi_parse(data_dir)
  else:
    parsed_data = parse(data_dir)

  data = {}

  ### normalize input
  symbol_counter = Counter()
  for utterance in tqdm(parsed_data):
    chn = utterance[4]
    try:
      chn_ipa = chn_to_ipa(chn, add_period=True)
    except Exception as e:
      print("Error on:", chn, e)
      continue

    speaker_name = [1]
    basename = [2]
    wav_path = [3]

    text_symbols = extract_from_sentence(chn_ipa, ignore_tones, ignore_arcs)
    if speaker_name not in data:
      data[speaker_name] = []

    symbol_counter.update(text_symbols)
    data[speaker_name].append((basename, chn, chn_ipa, text_symbols, wav_path))

  all_symbols = OrderedDict(symbol_counter.most_common())
  all_symbols_path = get_all_symbols_path(base_dir, ds_name)
  save_json(all_symbols_path, all_symbols)

  all_speakers = [(k, len(v)) for k, v in data.items()]
  all_speakers.sort(key=lambda tup: tup[1], reverse=True)
  all_speakers = OrderedDict(all_speakers)
  all_speakers_path = get_all_speakers_path(base_dir, ds_name)
  save_json(all_speakers_path, all_speakers)

  for speaker, recordings in tqdm(data.items()):
    print("Processing speaker:", speaker)
    ### get all symbols
    symbols = set()
    for _, _, _, symbs, _ in recordings:
      current_symbols = set(symbs)
      #print(current_symbols)
      symbols = symbols.union(current_symbols)

    conv = init_from_symbols(symbols)

    ds_dir = get_ds_dir(base_dir, ds_name, speaker, create=True)

    conv.dump(os.path.join(ds_dir, ds_preprocessed_symbols_name))
    print("Resulting symbolset:")
    conv.print_symbols()

    ### convert text to symbols
    result = []
    for bn, chinese, chinese_ipa, syms, wav in recordings:
      symbol_ids = conv.symbols_to_ids(syms, add_eos=True, replace_unknown_with_pad=True)
      serialized_symbol_ids = serialize_symbol_ids(symbol_ids)
      duration = librosa.get_duration(filename=wav)
      symbols_str = ''.join(syms)
      #result.append((bn, wav, py, ipa_txt, serialized_symbol_ids, symbols_str, duration))
      result.append((bn, wav, serialized_symbol_ids, duration, chinese, chinese_ipa, symbols_str))

    ### save
    #dest_filename = os.path.join(dataset_path, 'preprocessed.txt')

    df = pd.DataFrame(result)
    df.to_csv(os.path.join(ds_dir, ds_preprocessed_file_name), header=None, index=None, sep=csv_separator)
    print("Dataset saved.")
    #df1 = df.iloc[:, [1, 4, 0, 2, 3, 5, 6]]
    #df2 = df.iloc[:, []]
    #df2.to_csv(os.path.join(ds_dir, ds_preprocessed_file_log_name), header=None, index=None, sep=csv_separator)

  
  print("Dataset preprocessing finished.")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--base_dir', type=str, help='base directory')
  parser.add_argument('--data_dir', type=str, help='THCHS dataset directory')
  parser.add_argument('--data_conversion_dir', type=str, help='THCHS dataset directory')
  parser.add_argument('--ignore_tones', action='store_true')
  parser.add_argument('--ignore_arcs', action='store_true')
  parser.add_argument('--ds_name', type=str, help='the name you want to call the dataset')
  parser.add_argument('--auto_dl', action='store_true')
  parser.add_argument('--auto_convert', action='store_true')
  parser.add_argument('--no_debugging', action='store_true')
  parser.add_argument('--kaldi_version', action='store_true')

  args = parser.parse_args()

  if not args.no_debugging:
    args.base_dir = '/datasets/models/taco2pt_v2'
    args.data_dir = '/datasets/thchs_wav'
    args.data_conversion_dir = '/datasets/thchs_16bit_22050kHz'
    args.ds_name = 'thchs_v5-test'
    args.ignore_tones = True
    args.ignore_arcs = True
    args.auto_dl = False
    args.auto_convert = False
    args.kaldi_version = False
  
  if args.auto_dl:
    if args.kaldi_version:
      kaldi_ensure_downloaded(args.data_dir)
    else:
      ensure_downloaded(args.data_dir)

  if args.auto_convert:
    ensure_is_22050kHz(args.data_dir, args.data_conversion_dir, args.kaldi_version)

  preprocess(args.base_dir, args.data_conversion_dir, args.ds_name, args.ignore_tones, args.ignore_arcs, args.kaldi_version)
