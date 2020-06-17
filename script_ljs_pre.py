import argparse
import os
from parser.LJSpeechDatasetParser import LJSpeechDatasetParser

import epitran
import pandas as pd
from tqdm import tqdm

from ipa2symb import extract_from_sentence
from paths import get_ds_dir, ds_preprocessed_file_log_name, ds_preprocessed_file_name, ds_preprocessed_symbols_log_name, ds_preprocessed_symbols_name
from text.adjustments import normalize_text
from text.symbol_converter import init_from_symbols, serialize_symbol_ids
from utils import csv_separator

def preprocess(base_dir: str, data_dir: str, ds_name: str, ipa: bool):
  epi = epitran.Epitran('eng-Latn')
  p = LJSpeechDatasetParser(data_dir)
  p.parse()

  speaker = "1"

  ds_dir = get_ds_dir(base_dir, ds_name, speaker, create=True)
  
  data = []

  ### normalize input
  for basename, text, wav_path in tqdm(p.data):
    normalized_text = normalize_text(text)
    if use_ipa:
      ipa_text = epi.transliterate(normalized_text)
      text_symbols = extract_from_sentence(ipa_text)
    else:
      ipa_text = ''
      text_symbols = list(normalized_text)
    data.append((basename, normalized_text, ipa_text, text_symbols, wav_path))

  ### get all symbols
  symbols = set()
  for _, _, _, symbs, _ in data:
    current_symbols = set(symbs)
    #print(current_symbols)
    symbols = symbols.union(current_symbols)
  conv = init_from_symbols(symbols)

  conv.dump(os.path.join(ds_dir, ds_preprocessed_symbols_name))
  conv.plot(os.path.join(ds_dir, ds_preprocessed_symbols_log_name))
  print('Resulting symbolset:')
  conv.print_symbols()

  ### convert text to symbols
  result = []
  for bn, norm_text, ipa_txt, syms, wav in data:
    symbol_ids = conv.symbols_to_ids(syms, add_eos=True, replace_unknown_with_pad=True)
    serialized_symbol_ids = serialize_symbol_ids(symbol_ids)
    result.append((bn, wav, norm_text, ipa_txt, serialized_symbol_ids))

  ### save
  #dest_filename = os.path.join(dataset_path, 'preprocessed.txt')
 
  df = pd.DataFrame(result)
  df1 = df.iloc[:, [1, 4]]
  df1.to_csv(os.path.join(ds_dir, ds_preprocessed_file_name), header=None, index=None, sep=csv_separator)
  print("Dataset saved.")
  df2 = df.iloc[:, [0, 2, 3]]
  df2.to_csv(os.path.join(ds_dir, ds_preprocessed_file_log_name), header=None, index=None, sep=csv_separator)
  print("Dataset preprocessing finished.")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--base_dir', type=str, help='base directory')
  parser.add_argument('--data_dir', type=str, help='LJSpeech dataset directory')
  parser.add_argument('--ipa', type=str, help='transcribe to IPA')
  parser.add_argument('--ds_name', type=str, help='the name you want to call the dataset')
  parser.add_argument('--debug', type=str, default="true")

  args = parser.parse_args()

  debug = str.lower(args.debug) == 'true'

  if debug:
    args.base_dir = '/datasets/models/taco2pt_v2'
    args.data_dir = '/datasets/LJSpeech-1.1'
    args.ipa = 'false'
    args.ds_name = 'ljs_en'
  
  use_ipa = str.lower(args.ipa) == 'true'

  preprocess(args.base_dir, args.data_dir, args.ds_name, use_ipa)

  