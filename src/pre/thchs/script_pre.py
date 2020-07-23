import argparse
import os
import shutil
import tarfile
import tempfile
from collections import Counter, OrderedDict
from pathlib import Path

import epitran
import librosa
import pandas as pd
import wget
from tqdm import tqdm

from src.common.utils import csv_separator, download_tar, save_json
from src.parser.thchs_kaldi_parser import \
    ensure_downloaded as kaldi_ensure_downloaded
from src.parser.thchs_kaldi_parser import exists as kaldi_exists
from src.parser.thchs_kaldi_parser import parse as kaldi_parse
from src.parser.thchs_parser import ensure_downloaded, exists, parse
from src.script_paths import (ds_preprocessed_file_name,
                              ds_preprocessed_symbols_name,
                              get_all_speakers_path, get_all_symbols_path,
                              get_ds_dir)
from src.text.adjustments import normalize_text
from src.text.chn_tools import chn_to_ipa
from src.text.ipa2symb import extract_from_sentence
from src.text.symbol_converter import init_from_symbols, serialize_symbol_ids

def preprocess(base_dir: str, data_dir: str, ds_name: str, ignore_tones: bool, ignore_arcs: bool, kaldi_version: bool):
  all_symbols_path = get_all_symbols_path(base_dir, ds_name)
  all_speakers_path = get_all_speakers_path(base_dir, ds_name)
  
  already_preprocessed = os.path.exists(all_speakers_path) and os.path.exists(all_symbols_path)
  if already_preprocessed:
    print("Data is already preprocessed for dataset: {}".format(ds_name))
    return

  if kaldi_version:
    parsed_data = kaldi_parse(data_dir)
  else:
    parsed_data = parse(data_dir)

  data = {}

  ### normalize input
  symbol_counter = Counter()
  print("Converting Chinese to IPA.")
  for utterance in tqdm(parsed_data):
    chn = utterance[4]
    try:
      chn_ipa = chn_to_ipa(chn, add_period=True)
    except Exception as e:
      print("Error on:", chn, e)
      continue

    speaker_name = utterance[1]
    basename = utterance[2]
    wav_path = utterance[3]

    text_symbols = extract_from_sentence(chn_ipa, ignore_tones, ignore_arcs)
    if speaker_name not in data:
      data[speaker_name] = []

    symbol_counter.update(text_symbols)
    data[speaker_name].append((basename, chn, chn_ipa, text_symbols, wav_path))

  all_symbols = OrderedDict(symbol_counter.most_common())
  save_json(all_symbols_path, all_symbols)

  all_speakers = [(k, len(v)) for k, v in data.items()]
  all_speakers.sort(key=lambda tup: tup[1], reverse=True)
  all_speakers = OrderedDict(all_speakers)
  save_json(all_speakers_path, all_speakers)
  print("Done.")

  print("Reading wav durations and processing symbols.")
  for speaker, recordings in tqdm(data.items()):
    #print("Processing speaker:", speaker)
    ### get all symbols
    symbols = set()
    for _, _, _, symbs, _ in recordings:
      current_symbols = set(symbs)
      #print(current_symbols)
      symbols = symbols.union(current_symbols)

    conv = init_from_symbols(symbols)

    ds_dir = get_ds_dir(base_dir, ds_name, speaker, create=True)

    conv.dump(os.path.join(ds_dir, ds_preprocessed_symbols_name))
    #print("Resulting symbolset:")
    #conv.print_symbols()

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
    #print("Dataset saved.")
    #df1 = df.iloc[:, [1, 4, 0, 2, 3, 5, 6]]
    #df2 = df.iloc[:, []]
    #df2.to_csv(os.path.join(ds_dir, ds_preprocessed_file_log_name), header=None, index=None, sep=csv_separator)
  print("Done.")
  print("Dataset preprocessing finished.")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--base_dir', type=str, help='base directory')
  parser.add_argument('--no_debugging', action='store_true')
  parser.add_argument('--kaldi_version', action='store_true')
  parser.add_argument('--data_dir', type=str, help='THCHS dataset directory')

  parser.add_argument('--ignore_tones', action='store_true')
  parser.add_argument('--ignore_arcs', action='store_true')
  parser.add_argument('--ds_name', type=str, help='the name you want to call the dataset')

  args = parser.parse_args()
  
  if not args.no_debugging:
    args.kaldi_version = True
    args.kaldi_version = False
    args.base_dir = '/datasets/models/taco2pt_v2'
    if args.kaldi_version:
      args.data_dir = '/datasets/THCHS-30'
      args.ds_name = 'thchs_kaldi_v5-test'
      args.ignore_tones = True
      args.ignore_arcs = True
    else:
      args.data_dir = '/datasets/thchs_wav'
      args.ds_name = 'thchs_v5-test'
      args.ignore_tones = True
      args.ignore_arcs = True
  
  # TODO: check already exists
  
  preprocess(args.base_dir, args.data_dir, args.ds_name, args.ignore_tones, args.ignore_arcs, args.kaldi_version)
