import os
from src.parser.LJSpeechDatasetParser import LJSpeechDatasetParser, get_metadata_filepath
import wget
import tarfile
import shutil

import epitran
import pandas as pd
from tqdm import tqdm
import librosa

from src.text.ipa2symb import extract_from_sentence
from src.paths import get_ds_dir, ds_preprocessed_file_name, ds_preprocessed_symbols_name, get_all_symbols_path, get_all_speakers_path
from src.text.adjustments import normalize_text
from src.text.symbol_converter import init_from_symbols, serialize_symbol_ids
from src.common.utils import csv_separator
from collections import Counter, OrderedDict
from src.common.utils import save_json

def __download_ljs(dir_path: str):
  print("LJSpeech is not downloaded yet.")
  print("Starting download...")
  os.makedirs(dir_path, exist_ok=True)
  download_url = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
  dest = wget.download(download_url, dir_path)
  downloaded_file = os.path.join(dir_path, dest)
  print("\nFinished download to {}".format(downloaded_file))
  print("Unpacking...")
  tar = tarfile.open(downloaded_file, "r:bz2")  
  tar.extractall(dir_path)
  tar.close()
  print("Done.")
  print("Moving files...")
  dir_name = "LJSpeech-1.1"
  ljs_data_dir = os.path.join(dir_path, dir_name)
  files = os.listdir(ljs_data_dir)
  for f in tqdm(files):
    shutil.move(os.path.join(ljs_data_dir, f), dir_path)
  print("Done.")
  os.remove(downloaded_file)
  os.rmdir(ljs_data_dir)

def ensure_downloaded(dir_path: str):
  metadata_filepath = get_metadata_filepath(dir_path)
  metadata_file_exists = os.path.exists(metadata_filepath)
  if not metadata_file_exists:
    __download_ljs(dir_path)

def preprocess(base_dir: str, data_dir: str, ds_name: str, ipa: bool, ignore_arcs: bool):
  all_symbols_path = get_all_symbols_path(base_dir, ds_name)
  all_speakers_path = get_all_speakers_path(base_dir, ds_name)
  
  already_preprocessed = os.path.exists(all_speakers_path) and os.path.exists(all_symbols_path)
  if already_preprocessed:
    print("Data is already preprocessed for dataset: {}".format(ds_name))
    return

  epi = epitran.Epitran('eng-Latn')
  p = LJSpeechDatasetParser(data_dir)
  p.parse()

  speaker = "1"

  ds_dir = get_ds_dir(base_dir, ds_name, speaker, create=True)

  all_speakers = OrderedDict([("1", len(p.data))])
  save_json(all_speakers_path, all_speakers)

  data = []
  symbol_counter = Counter()
  print("Reading symbols...")
  # Duration: ~20min
  ### normalize input
  for basename, text, wav_path in tqdm(p.data):
    normalized_text = normalize_text(text)
    if ipa:
      ipa_text = epi.transliterate(normalized_text)
      text_symbols = extract_from_sentence(ipa_text, ignore_tones=False, ignore_arcs=ignore_arcs)
      data.append((basename, normalized_text, ipa_text, text_symbols, wav_path))
    else:
      text_symbols = list(normalized_text)
      data.append((basename, normalized_text, normalized_text, text_symbols, wav_path))
    symbol_counter.update(text_symbols)

  all_symbols = OrderedDict(symbol_counter.most_common())
  save_json(all_symbols_path, all_symbols)

  ### get all symbols
  symbols = set()
  for _, _, _, symbs, _ in data:
    current_symbols = set(symbs)
    #print(current_symbols)
    symbols = symbols.union(current_symbols)
  conv = init_from_symbols(symbols)

  conv.dump(os.path.join(ds_dir, ds_preprocessed_symbols_name))
  print('Resulting symbolset:')
  conv.print_symbols()

  ### convert text to symbols
  # Duration: ~15min
  print("Reading durations of audio files...")
  result = []
  for bn, norm_eng, eng_ipa, syms, wav in tqdm(data):
    symbol_ids = conv.symbols_to_ids(syms, add_eos=True, replace_unknown_with_pad=True)
    serialized_symbol_ids = serialize_symbol_ids(symbol_ids)
    duration = librosa.get_duration(filename=wav)
    symbols_str = ''.join(syms)
    result.append((bn, wav, serialized_symbol_ids, duration, norm_eng, eng_ipa, symbols_str))
    #result.append((bn, wav, norm_text, ipa_txt, serialized_symbol_ids, duration))

  ### save
  #dest_filename = os.path.join(dataset_path, 'preprocessed.txt')
 
  df = pd.DataFrame(result)
  df.to_csv(os.path.join(ds_dir, ds_preprocessed_file_name), header=None, index=None, sep=csv_separator)
  print("Dataset saved.")

  # df = pd.DataFrame(result)
  # df1 = df.iloc[:, [1, 4]]
  # df1.to_csv(os.path.join(ds_dir, ds_preprocessed_file_name), header=None, index=None, sep=csv_separator)
  # print("Dataset saved.")
  # df2 = df.iloc[:, [0, 2, 3, 5]]
  # df2.to_csv(os.path.join(ds_dir, ds_preprocessed_file_log_name), header=None, index=None, sep=csv_separator)
  print("Dataset preprocessing finished.")

def main(base_dir: str, data_dir: str, ipa: bool, ignore_arcs: bool, ds_name: str, auto_dl: bool):
  if auto_dl:
    ensure_downloaded(data_dir)

  preprocess(base_dir, data_dir, ds_name, ipa, ignore_arcs)

if __name__ == "__main__":
  main(
    base_dir = '/datasets/models/taco2pt_v2-tmp',
    data_dir = '/datasets/LJSpeech-1.1-tmp',
    ipa = True,
    ignore_arcs = True,
    ds_name = 'ljs_ipa_v2-tmp'
  )