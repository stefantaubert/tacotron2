import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from paths import preprocessed_file_name, test_file_name, training_file_name, validation_file_name, filelist_dir, symbols_path_name, symbols_path_info_name
from text.conversion.SymbolConverter import get_from_file, get_from_symbols, _eos, _pad, get_symbols_from_str
from utils import csv_separator

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('--base_dir', type=str, help='base directory', default='/datasets/models/taco2pt_ms_learning')
  parser.add_argument('--ds_name', default='thchs')
  parser.add_argument('--speakers', default='A11,C17,B7')
  
  args = parser.parse_args()
  filelist_dir_path = os.path.join(args.base_dir, filelist_dir)

  speakers = args.speakers.split(',')
  tmp = []
  for i, speaker in enumerate(speakers):
    speaker_dir = os.path.join(filelist_dir_path, args.ds_name, speaker)
    prepr_path = os.path.join(speaker_dir, preprocessed_file_name)
    speaker_conv = get_from_file(os.path.join(filelist_dir_path, args.ds_name, speaker, symbols_path_name))

    data = pd.read_csv(prepr_path, header=None, sep=csv_separator)
   
    tmp.append((i, speaker_conv, data))

  conc = None
  symbols = set()
  for _, speaker_conv, _ in tmp:
    symbols = symbols.union(speaker_conv.get_symbols())
  symbols.remove(_eos)
  symbols.remove(_pad)
  new_conv = get_from_symbols(symbols)

  result = []
  #df = pd.DataFrame()
  for speaker_id, speaker_conv, data in tmp:
    for i, row in data.iterrows():
      symb_seq = row[1]
      symb_seq_int = get_symbols_from_str(symb_seq)
      original_symbols = speaker_conv.sequence_to_original_chars(symb_seq_int)
      new_symb_seq = new_conv.text_to_sequence(original_symbols)
      seq_str = ",".join([str(s) for s in new_symb_seq])
      result.append((row[0], seq_str, speaker_id))
  new_conv.dump(os.path.join(filelist_dir_path, symbols_path_name))
  new_conv.plot(os.path.join(filelist_dir_path, symbols_path_info_name))
  df = pd.DataFrame(result)
  print(df.head())
  df.to_csv(os.path.join(filelist_dir_path, preprocessed_file_name), header=None, index=None, sep=csv_separator)
  print("Done.")