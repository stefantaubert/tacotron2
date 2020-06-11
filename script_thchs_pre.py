import argparse
import os
from parser.thchs_parser import parse as parse_thchs
from dragonmapper import hanzi

import epitran
import pandas as pd
from tqdm import tqdm

from ipa2symb import extract_from_sentence
from paths import preprocessed_file_name, preprocessed_file_debug_name, symbols_path_name, symbols_path_info_name, filelist_dir
from text.adjustments import normalize_text
from text.conversion.SymbolConverter import get_from_symbols
from utils import csv_separator

def chn_to_ipa(chn):
  chn_words = chn.split(' ')
  res = []
  for w in chn_words:
    chn_syll_ipa = hanzi.to_ipa(w)
    chn_ipa = chn_syll_ipa.replace(' ', '')
    res.append(chn_ipa)
  res = ' '.join(res)
  return res

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--base_dir', type=str, help='base directory', default='/datasets/models/taco2pt_ms_learning')
  parser.add_argument('--data_dir', type=str, help='THCHS dataset directory', default='/datasets/thchs_wav')
  parser.add_argument('--ds_name', type=str, help='the name you want to call the dataset', default='thchs')
  parser.add_argument('--ignore_tones', type=str, default='true')
  #parser.add_argument('--id', type=str, help='id which should be assigned to the dataset', default='2')

  args = parser.parse_args()
  #use_ipa = str.lower(args.ipa) == 'true'
  ignore_tones = str.lower(args.ignore_tones) == 'true'

  parsed_data = parse_thchs(args.data_dir)
  data = {}

  ### normalize input
  for _, speaker_name, basename, wav_path, chn in tqdm(parsed_data):
    try:
      chn_ipa = chn_to_ipa(chn)
    except Exception as e:
      # print("Error on:", x, e)
      continue

    text_symbols = extract_from_sentence(chn_ipa, ignore_tones)
    if speaker_name not in data:
      data[speaker_name] = []

    data[speaker_name].append((basename, chn, chn_ipa, text_symbols, wav_path))

  for speaker, recordings in data.items():
    print("Processing speaker:", speaker)
    ### get all symbols
    symbols = set()
    for _, _, _, symbs, _ in recordings:
      current_symbols = set(symbs)
      #print(current_symbols)
      symbols = symbols.union(current_symbols)

    conv = get_from_symbols(symbols)

    speaker_dir = os.path.join(args.base_dir, filelist_dir, args.ds_name, speaker)
    os.makedirs(speaker_dir, exist_ok=True)

    conv.dump(os.path.join(speaker_dir, symbols_path_name))
    conv.plot(os.path.join(speaker_dir, symbols_path_info_name))
    print(conv.get_symbols())

    ### convert text to symbols
    result = []
    for bn, py, ipa_txt, sym, wav in recordings:
      sym_str = ''.join(sym)
      seq = conv.text_to_sequence(sym)
      seq_str = ",".join([str(s) for s in seq])
      result.append((bn, wav, py, ipa_txt, seq_str, speaker, sym_str))

    ### save
    #dest_filename = os.path.join(dataset_path, 'preprocessed.txt')

    df = pd.DataFrame(result)
    df1 = df.iloc[:, [1, 4]]
    df1.to_csv(os.path.join(speaker_dir, preprocessed_file_name), header=None, index=None, sep=csv_separator)
    print("Dataset saved.")
    df2 = df.iloc[:, [0, 2, 3, 6]]
    df2.to_csv(os.path.join(speaker_dir, preprocessed_file_debug_name), header=None, index=None, sep=csv_separator)
    print("Dataset preprocessing finished.")
