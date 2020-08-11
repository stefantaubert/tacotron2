import os
from src.core.pre.text import TextDataList, SymbolsDict
from src.core.pre.ds import DsDataList, DsData
from src.cli.pre.paths import get_text_subdir, get_text_csv, get_text_symbols_json, get_text_symbol_converter
from src.cli.pre.ds import load_ds_csv
from src.text.symbol_converter import SymbolConverter
from argparse import ArgumentParser
from src.core.pre.text import normalize as normalize_core
from src.core.pre.text import preprocess as preprocess_core
from src.core.pre.text import convert_to_ipa as convert_to_ipa_core

#region IO

def load_text_symbol_converter(base_dir: str, ds_name: str, sub_name: str) -> SymbolConverter:
  data_path = get_text_symbol_converter(base_dir, ds_name, sub_name)
  return SymbolConverter.load_from_file(data_path)
  
def save_text_symbol_converter(base_dir: str, ds_name: str, sub_name: str, data: SymbolConverter):
  data_path = get_text_symbol_converter(base_dir, ds_name, sub_name)
  data.dump(data_path)

def load_text_symbols_json(base_dir: str, ds_name: str, sub_name: str) -> SymbolsDict:
  data_path = get_text_symbols_json(base_dir, ds_name, sub_name)
  return SymbolsDict.load(data_path)
  
def save_text_symbols_json(base_dir: str, ds_name: str, sub_name: str, data: SymbolsDict):
  data_path = get_text_symbols_json(base_dir, ds_name, sub_name)
  data.save(data_path)

def load_text_csv(base_dir: str, ds_name: str, sub_name: str) -> TextDataList:
  origin_data_path = get_text_csv(base_dir, ds_name, sub_name)
  return TextDataList.load(origin_data_path)
  
def save_text_csv(base_dir: str, ds_name: str, sub_name: str, data: TextDataList):
  data_path = get_text_csv(base_dir, ds_name, sub_name)
  data.save(data_path)

def text_subdir_exists(base_dir: str, ds_name: str, sub_name: str):
  data_dir = get_text_subdir(base_dir, ds_name, sub_name, create=False)
  return os.path.exists(data_dir)

#endregion

#region Processing text

def init_pre_parser(parser: ArgumentParser):
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--ds_name', type=str, required=True)
  parser.add_argument('--sub_name', type=str, required=True)
  return preprocess

def preprocess(base_dir: str, ds_name: str, sub_name: str):
  if text_subdir_exists(base_dir, ds_name, sub_name):
    print("Already exists.")
  else:
    data = load_ds_csv(base_dir, ds_name)
    #wav_data_dir = get_pre_ds_wav_subname_dir(base_dir, ds_name, sub_name, create=False)
    text_data, conv, all_symbols = preprocess_core(data)
    save_text_csv(base_dir, ds_name, sub_name, text_data)
    save_text_symbol_converter(base_dir, ds_name, sub_name, conv)
    save_text_symbols_json(base_dir, ds_name, sub_name, all_symbols)

#endregion

#region Normalizing

def init_normalize_parser(parser: ArgumentParser):
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--ds_name', type=str, required=True)
  parser.add_argument('--origin_sub_name', type=str, required=True)
  parser.add_argument('--destination_sub_name', type=str, required=True)
  return normalize

def normalize(base_dir: str, ds_name: str, origin_sub_name: str, destination_sub_name: str):
  if text_subdir_exists(base_dir, ds_name, destination_sub_name):
    print("Already exists.")
  else:
    data = load_text_csv(base_dir, ds_name, origin_sub_name)
    #wav_data_dir = get_pre_ds_wav_subname_dir(base_dir, ds_name, sub_name, create=False)
    text_data, conv, all_symbols = normalize_core(data)
    save_text_csv(base_dir, ds_name, destination_sub_name, text_data)
    save_text_symbol_converter(base_dir, ds_name, destination_sub_name, conv)
    save_text_symbols_json(base_dir, ds_name, destination_sub_name, all_symbols)

#endregion

#region IPA

def init_ipa_parser(parser: ArgumentParser):
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--ds_name', type=str, required=True)
  parser.add_argument('--origin_sub_name', type=str, required=True)
  parser.add_argument('--destination_sub_name', type=str, required=True)
  parser.add_argument('--ignore_tones', action='store_true')
  parser.add_argument('--ignore_arcs', action='store_true')
  return convert_to_ipa

def convert_to_ipa(base_dir: str, ds_name: str, origin_sub_name: str, destination_sub_name: str, ignore_tones: bool, ignore_arcs: bool):
  if text_subdir_exists(base_dir, ds_name, destination_sub_name):
    print("Already exists.")
  else:
    data = load_text_csv(base_dir, ds_name, origin_sub_name)
    #wav_data_dir = get_pre_ds_wav_subname_dir(base_dir, ds_name, sub_name, create=False)
    text_data, conv, all_symbols = convert_to_ipa_core(data, ignore_tones, ignore_arcs)
    save_text_csv(base_dir, ds_name, destination_sub_name, text_data)
    save_text_symbol_converter(base_dir, ds_name, destination_sub_name, conv)
    save_text_symbols_json(base_dir, ds_name, destination_sub_name, all_symbols)

#endregion


if __name__ == "__main__":
  preprocess(
    base_dir="/datasets/models/taco2pt_v2",
    ds_name="thchs",
    sub_name="chn",
  )

  convert_to_ipa(
    base_dir="/datasets/models/taco2pt_v2",
    ds_name="thchs",
    origin_sub_name="chn",
    destination_sub_name="ipa",
    ignore_tones=False,
    ignore_arcs=True,
  )

  preprocess(
    base_dir="/datasets/models/taco2pt_v2",
    ds_name="ljs",
    sub_name="en",
  )

  normalize(
    base_dir="/datasets/models/taco2pt_v2",
    ds_name="ljs",
    origin_sub_name="en",
    destination_sub_name="en_norm",
  )

  convert_to_ipa(
    base_dir="/datasets/models/taco2pt_v2",
    ds_name="ljs",
    origin_sub_name="en_norm",
    destination_sub_name="ipa_norm",
    ignore_tones=True,
    ignore_arcs=True,
  )