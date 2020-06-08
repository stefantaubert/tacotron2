import argparse
import os
from parser.thchs_parser import parse as parse_thchs
from dragonmapper import hanzi

import epitran
import pandas as pd
from tqdm import tqdm

from ipa2symb import extract_from_sentence
from paths import preprocessed_file, preprocessed_file_debug, symbols_path, symbols_path_info
from text.adjustments import normalize_text
from text.conversion.SymbolConverter import get_from_symbols

csv_separator = '\t'


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--base_dir', type=str, help='base directory', default='/datasets/models/taco2pt_testing')
  parser.add_argument('--thchs', type=str, help='THCHS dataset directory', default='/datasets/thchs_wav')
  #parser.add_argument('--id', type=str, help='id which should be assigned to the dataset', default='2')

  args = parser.parse_args()
  #use_ipa = str.lower(args.ipa) == 'true'

  parsed_data = parse_thchs(args.thchs)
  data = []

  ### normalize input
  for _, dir_name, basename, wav_path, pinyin in tqdm(parsed_data):
    try:
      pinyin_ipa = hanzi.pinyin_to_ipa(pinyin)
    except Exception as e:
      # print("Error on:", x, e)
      continue

    text_symbols = extract_from_sentence(pinyin_ipa)
    
    data.append((basename, pinyin, pinyin_ipa, text_symbols, wav_path))

  ### get all symbols
  symbols = set()
  for _, _, _, symbs, _ in data:
    current_symbols = set(symbs)
    #print(current_symbols)
    symbols = symbols.union(current_symbols)
  conv = get_from_symbols(symbols)
  conv.dump(os.path.join(args.base_dir, symbols_path))
  conv.plot(os.path.join(args.base_dir, symbols_path_info))
  print(conv.get_symbols())

  ### convert text to symbols
  result = []
  for bn, py, ipa_txt, sym, wav in data:
    seq = conv.text_to_sequence(sym)
    seq_str = ",".join([str(s) for s in seq])
    result.append((bn, wav, py, ipa_txt, seq_str))

  ### save
  #dest_filename = os.path.join(dataset_path, 'preprocessed.txt')

  df = pd.DataFrame(result)
  df1 = df.iloc[:, [1, 4]]
  df1.to_csv(os.path.join(args.base_dir, preprocessed_file), header=None, index=None, sep=csv_separator)
  print("Dataset saved.")
  df2 = df.iloc[:, [0, 2, 3]]
  df2.to_csv(os.path.join(args.base_dir, preprocessed_file_debug), header=None, index=None, sep=csv_separator)
  print("Dataset preprocessing finished.")
