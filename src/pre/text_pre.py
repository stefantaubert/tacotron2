import os
from collections import Counter, OrderedDict

import epitran
from tqdm import tqdm

from src.common.utils import load_csv, save_csv, save_json
from src.paths import (ds_preprocessed_file_name, ds_preprocessed_symbols_name,
                       get_all_speakers_path, get_all_symbols_path, get_ds_dir)
from src.pre.calc_mels import parse_data
from src.text.adjustments import normalize_text
from src.text.chn_tools import chn_to_ipa
from src.text.ipa2symb import extract_from_sentence
from src.text.symbol_converter import init_from_symbols, serialize_symbol_ids



def init_thchs_text_parser(parser):
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--mel_name', type=str, required=True)
  parser.add_argument('--ds_name', type=str, help='the name you want to call the dataset', required=True)
  parser.add_argument('--ignore_tones', action='store_true')
  parser.set_defaults(ignore_arcs=True, lang="chn", convert_to_ipa=True)
  return preprocess

def init_ljs_text_pre_parser(parser):
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--mel_name', type=str, required=True)
  parser.add_argument('--ds_name', type=str, help='the name you want to call the dataset', required=True)
  parser.add_argument('--convert_to_ipa', action='store_true', help='transcribe to IPA')
  parser.set_defaults(ignore_tones=True, ignore_arcs=True, lang="eng")
  return preprocess

def preprocess(base_dir: str, mel_name: str, ds_name: str, ignore_tones: bool, ignore_arcs: bool, lang: str, convert_to_ipa: bool):
  all_symbols_path = get_all_symbols_path(base_dir, ds_name)
  all_speakers_path = get_all_speakers_path(base_dir, ds_name)
  
  already_preprocessed = os.path.exists(all_speakers_path) and os.path.exists(all_symbols_path)
  if already_preprocessed:
    print("Data is already preprocessed for dataset: {}".format(ds_name))
    return

  data = {}

  if convert_to_ipa and lang == 'eng':
    epi = epitran.Epitran('eng-Latn')

  ### normalize input
  symbol_counter = Counter()
  if convert_to_ipa:
    print("Converting text to IPA...")
  else:
    print("Processing text...")
  parsed_data = parse_data(base_dir, mel_name)
  for basename, speaker_name, text, mel_path, duration in tqdm(parsed_data):
    if lang == "chn":
      if convert_to_ipa:
        try:
          ipa = chn_to_ipa(text, add_period=True)
        except Exception as e:
          print("Error on:", text, e)
          continue
    elif lang == "eng":
      text = normalize_text(text)

      if convert_to_ipa:
        ipa = epi.transliterate(text)

    if convert_to_ipa:
      symbols = extract_from_sentence(ipa, ignore_tones, ignore_arcs)
    else:
      ipa = '-- no IPA --'
      symbols = list(text)

    if speaker_name not in data:
      data[speaker_name] = []

    symbol_counter.update(symbols)
    data[speaker_name].append((basename, text, ipa, symbols, mel_path, duration))

  all_symbols = OrderedDict(symbol_counter.most_common())
  save_json(all_symbols_path, all_symbols)

  all_speakers = [(k, len(v)) for k, v in data.items()]
  all_speakers.sort(key=lambda tup: tup[1], reverse=True)
  all_speakers = OrderedDict(all_speakers)
  save_json(all_speakers_path, all_speakers)
  print("Done.")

  print("Processing symbols.")
  for speaker, recordings in tqdm(data.items()):
    #print("Processing speaker:", speaker)
    ### get all symbols
    symbols = set()
    for recording in recordings:
      symbs = recording[3]
      current_symbols = set(symbs)
      symbols = symbols.union(current_symbols)

    conv = init_from_symbols(symbols)
    ds_dir = get_ds_dir(base_dir, ds_name, speaker, create=True)
    ds_symbols_path = os.path.join(ds_dir, ds_preprocessed_symbols_name)
    conv.dump(ds_symbols_path)

    ### convert text to symbols
    result = []
    for basename, text, ipa, symbols, mel_path, duration in recordings:
      symbol_ids = conv.symbols_to_ids(symbols, add_eos=True, replace_unknown_with_pad=True)
      serialized_symbol_ids = serialize_symbol_ids(symbol_ids)
      symbols_str = ''.join(symbols)
      #result.append((bn, wav, py, ipa_txt, serialized_symbol_ids, symbols_str, duration))
      result.append((basename, mel_path, serialized_symbol_ids, duration, text, ipa, symbols_str))

    save_csv(result, os.path.join(ds_dir, ds_preprocessed_file_name))

  print("Dataset preprocessing finished.")



if __name__ == "__main__":
  # preprocess(
  #   base_dir = '/datasets/models/taco2pt_v2',
  #   mel_name = 'thchs',
  #   ds_name = 'thchs_v6',
  #   ignore_arcs = True,
  #   ignore_tones = False,
  #   lang = "chn",
  #   convert_to_ipa=True
  # )

  preprocess(
    base_dir = '/datasets/models/taco2pt_v2',
    mel_name = 'ljs',
    ds_name = 'ljs_ipa_v3',
    ignore_arcs = True,
    ignore_tones = True,
    lang = "eng",
    convert_to_ipa=True
  )
