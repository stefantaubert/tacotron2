import argparse
import os
from parser.LJSpeechDatasetParser import LJSpeechDatasetParser

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
  parser.add_argument('-b', '--base_dir', type=str, help='base directory', default='/datasets/models/taco2pt_testing')
  parser.add_argument('-d', '--ljspeech', type=str, help='LJSpeech dataset directory', default='/datasets/LJSpeech-1.1')
  parser.add_argument('-i', '--ipa', type=str, help='transcribe to IPA', default='true')

  args = parser.parse_args()
  use_ipa = str.lower(args.ipa) == 'true'
  epi = epitran.Epitran('eng-Latn')
  p = LJSpeechDatasetParser(args.ljspeech)
  p.parse()

  #print(p.data)

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
  conv = get_from_symbols(symbols)
  conv.dump(os.path.join(args.base_dir, symbols_path))
  conv.plot(os.path.join(args.base_dir, symbols_path_info))
  print(conv.get_symbols())

  ### convert text to symbols
  result = []
  for bn, norm_text, ipa_txt, sym, wav in data:
    seq = conv.text_to_sequence(sym)
    seq_str = ",".join([str(s) for s in seq])
    result.append((bn, wav, norm_text, ipa_txt, seq_str))

  ### save
  #dest_filename = os.path.join(dataset_path, 'preprocessed.txt')

  df = pd.DataFrame(result)
  df1 = df.iloc[:, [1, 4]]
  df1.to_csv(os.path.join(args.base_dir, preprocessed_file), header=None, index=None, sep=csv_separator)
  print("Dataset saved.")
  df2 = df.iloc[:, [0, 2, 3]]
  df2.to_csv(os.path.join(args.base_dir, preprocessed_file_debug), header=None, index=None, sep=csv_separator)
  print("Dataset preprocessing finished.")
