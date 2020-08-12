import os
from argparse import ArgumentParser

from src.cli.pre.ds import load_ds_csv
from src.cli.pre.paths import (get_text_csv, get_text_subdir,
                               get_text_symbol_converter,
                               get_text_symbols_json)
from src.core.pre import (DsData, DsDataList, SymbolsDict, TextDataList,
                          text_convert_to_ipa, text_normalize, text_preprocess)
from src.text.symbol_converter import SymbolConverter

#region IO

def load_text_symbol_converter(base_dir: str, ds_name: str, sub_name: str) -> SymbolConverter:
  data_path = get_text_symbol_converter(base_dir, ds_name, sub_name)
  return SymbolConverter.load_from_file(data_path)
  
def _save_text_symbol_converter(base_dir: str, ds_name: str, sub_name: str, data: SymbolConverter):
  data_path = get_text_symbol_converter(base_dir, ds_name, sub_name)
  data.dump(data_path)

def load_text_symbols_json(base_dir: str, ds_name: str, sub_name: str) -> SymbolsDict:
  data_path = get_text_symbols_json(base_dir, ds_name, sub_name)
  return SymbolsDict.load(data_path)
  
def _save_text_symbols_json(base_dir: str, ds_name: str, sub_name: str, data: SymbolsDict):
  data_path = get_text_symbols_json(base_dir, ds_name, sub_name)
  data.save(data_path)

def load_text_csv(base_dir: str, ds_name: str, sub_name: str) -> TextDataList:
  origin_data_path = get_text_csv(base_dir, ds_name, sub_name)
  return TextDataList.load(origin_data_path)
  
def _save_text_csv(base_dir: str, ds_name: str, sub_name: str, data: TextDataList):
  data_path = get_text_csv(base_dir, ds_name, sub_name)
  data.save(data_path)

def _text_subdir_exists(base_dir: str, ds_name: str, sub_name: str):
  data_dir = get_text_subdir(base_dir, ds_name, sub_name, create=False)
  return os.path.exists(data_dir)

#endregion

#region Processing text

def init_pre_parser(parser: ArgumentParser):
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--ds_name', type=str, required=True)
  parser.add_argument('--sub_name', type=str, required=True)
  return _preprocess

def _preprocess(base_dir: str, ds_name: str, sub_name: str):
  if _text_subdir_exists(base_dir, ds_name, sub_name):
    print("Already exists.")
  else:
    data = load_ds_csv(base_dir, ds_name)
    #wav_data_dir = get_pre_ds_wav_subname_dir(base_dir, ds_name, sub_name, create=False)
    text_data, conv, all_symbols = text_preprocess(data)
    _save_text_csv(base_dir, ds_name, sub_name, text_data)
    _save_text_symbol_converter(base_dir, ds_name, sub_name, conv)
    _save_text_symbols_json(base_dir, ds_name, sub_name, all_symbols)

#endregion

#region Normalizing

def init_normalize_parser(parser: ArgumentParser):
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--ds_name', type=str, required=True)
  parser.add_argument('--origin_sub_name', type=str, required=True)
  parser.add_argument('--destination_sub_name', type=str, required=True)
  return _normalize

def _normalize(base_dir: str, ds_name: str, origin_sub_name: str, destination_sub_name: str):
  if _text_subdir_exists(base_dir, ds_name, destination_sub_name):
    print("Already exists.")
  else:
    data = load_text_csv(base_dir, ds_name, origin_sub_name)
    #wav_data_dir = get_pre_ds_wav_subname_dir(base_dir, ds_name, sub_name, create=False)
    text_data, conv, all_symbols = text_normalize(data)
    _save_text_csv(base_dir, ds_name, destination_sub_name, text_data)
    _save_text_symbol_converter(base_dir, ds_name, destination_sub_name, conv)
    _save_text_symbols_json(base_dir, ds_name, destination_sub_name, all_symbols)

#endregion

#region IPA

def init_ipa_parser(parser: ArgumentParser):
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--ds_name', type=str, required=True)
  parser.add_argument('--origin_sub_name', type=str, required=True)
  parser.add_argument('--destination_sub_name', type=str, required=True)
  parser.add_argument('--ignore_tones', action='store_true')
  parser.add_argument('--ignore_arcs', action='store_true')
  return _convert_to_ipa

def _convert_to_ipa(base_dir: str, ds_name: str, origin_sub_name: str, destination_sub_name: str, ignore_tones: bool, ignore_arcs: bool):
  if _text_subdir_exists(base_dir, ds_name, destination_sub_name):
    print("Already exists.")
  else:
    data = load_text_csv(base_dir, ds_name, origin_sub_name)
    #wav_data_dir = get_pre_ds_wav_subname_dir(base_dir, ds_name, sub_name, create=False)
    text_data, conv, all_symbols = text_convert_to_ipa(data, ignore_tones, ignore_arcs)
    _save_text_csv(base_dir, ds_name, destination_sub_name, text_data)
    _save_text_symbol_converter(base_dir, ds_name, destination_sub_name, conv)
    _save_text_symbols_json(base_dir, ds_name, destination_sub_name, all_symbols)

#endregion


if __name__ == "__main__":
  _preprocess(
    base_dir="/datasets/models/taco2pt_v2",
    ds_name="thchs",
    sub_name="chn",
  )

  _convert_to_ipa(
    base_dir="/datasets/models/taco2pt_v2",
    ds_name="thchs",
    origin_sub_name="chn",
    destination_sub_name="ipa",
    ignore_tones=False,
    ignore_arcs=True,
  )

  _preprocess(
    base_dir="/datasets/models/taco2pt_v2",
    ds_name="ljs",
    sub_name="en",
  )

  _normalize(
    base_dir="/datasets/models/taco2pt_v2",
    ds_name="ljs",
    origin_sub_name="en",
    destination_sub_name="en_norm",
  )

  _convert_to_ipa(
    base_dir="/datasets/models/taco2pt_v2",
    ds_name="ljs",
    origin_sub_name="en_norm",
    destination_sub_name="ipa_norm",
    ignore_tones=True,
    ignore_arcs=True,
  )
