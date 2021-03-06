import os
from functools import partial

from src.app.pre.ds import get_ds_dir, load_ds_csv, load_symbols_json
from src.core.common.symbol_id_dict import SymbolIdDict
from src.core.common.utils import get_subdir
from src.core.pre.text.pre import SymbolsDict, TextData, TextDataList
from src.core.pre.text.pre import convert_to_ipa as text_convert_to_ipa_core
from src.core.pre.text.pre import normalize as text_normalize_core
from src.core.pre.text.pre import preprocess as text_preprocess_core

_text_data_csv = "data.csv"
_text_symbols_json = "symbols.json"
_text_symbol_ids_json = "symbol_ids.json"


def _get_text_root_dir(ds_dir: str, create: bool = False):
  return get_subdir(ds_dir, "text", create)


def get_text_dir(ds_dir: str, text_name: str, create: bool = False):
  return get_subdir(_get_text_root_dir(ds_dir, create), text_name, create)


def load_text_symbol_converter(text_dir: str) -> SymbolIdDict:
  path = os.path.join(text_dir, _text_symbol_ids_json)
  return SymbolIdDict.load_from_file(path)


def save_text_symbol_converter(text_dir: str, data: SymbolIdDict):
  path = os.path.join(text_dir, _text_symbol_ids_json)
  data.save(path)


def load_text_symbols_json(text_dir: str) -> SymbolsDict:
  path = os.path.join(text_dir, _text_symbols_json)
  return SymbolsDict.load(path)


def save_text_symbols_json(text_dir: str, data: SymbolsDict):
  path = os.path.join(text_dir, _text_symbols_json)
  data.save(path)


def load_text_csv(text_dir: str) -> TextDataList:
  path = os.path.join(text_dir, _text_data_csv)
  return TextDataList.load(TextData, path)


def save_text_csv(text_dir: str, data: TextDataList):
  path = os.path.join(text_dir, _text_data_csv)
  data.save(path)


def preprocess_text(base_dir: str, ds_name: str, text_name: str):
  print("Preprocessing text...")
  ds_dir = get_ds_dir(base_dir, ds_name)
  text_dir = get_text_dir(ds_dir, text_name)
  if os.path.isdir(text_dir):
    print("Already exists.")
  else:
    data = load_ds_csv(ds_dir)
    symbol_ids = load_symbols_json(ds_dir)
    text_data, conv, all_symbols = text_preprocess_core(data, symbol_ids)
    os.makedirs(text_dir)
    save_text_csv(text_dir, text_data)
    save_text_symbol_converter(text_dir, conv)
    save_text_symbols_json(text_dir, all_symbols)


def _text_op(base_dir: str, ds_name: str, orig_text_name: str, dest_text_name: str, operation):
  ds_dir = get_ds_dir(base_dir, ds_name)
  orig_text_dir = get_text_dir(ds_dir, orig_text_name)
  assert os.path.isdir(orig_text_dir)
  dest_text_dir = get_text_dir(ds_dir, dest_text_name)
  if os.path.isdir(dest_text_dir):
    print("Already exists.")
  else:
    print("Reading data...")
    data = load_text_csv(orig_text_dir)
    orig_conv = load_text_symbol_converter(orig_text_dir)
    text_data, conv, all_symbols = operation(data, orig_conv)
    os.makedirs(dest_text_dir)
    save_text_csv(dest_text_dir, text_data)
    save_text_symbol_converter(dest_text_dir, conv)
    save_text_symbols_json(dest_text_dir, all_symbols)
    print("Dataset processed.")


def text_normalize(base_dir: str, ds_name: str, orig_text_name: str, dest_text_name: str):
  print("Normalizing text...")
  operation = partial(text_normalize_core)
  _text_op(base_dir, ds_name, orig_text_name, dest_text_name, operation)


def text_convert_to_ipa(base_dir: str, ds_name: str, orig_text_name: str, dest_text_name: str, ignore_tones: bool = False, ignore_arcs: bool = False):
  print("Converting text to IPA...")
  operation = partial(text_convert_to_ipa_core, ignore_tones=ignore_tones,
                      ignore_arcs=ignore_arcs)
  _text_op(base_dir, ds_name, orig_text_name, dest_text_name, operation)


if __name__ == "__main__":
  preprocess_text(
    base_dir="/datasets/models/taco2pt_v5",
    ds_name="ljs",
    text_name="en",
  )

  text_normalize(
    base_dir="/datasets/models/taco2pt_v5",
    ds_name="ljs",
    orig_text_name="en",
    dest_text_name="en_norm",
  )

  text_convert_to_ipa(
    base_dir="/datasets/models/taco2pt_v5",
    ds_name="ljs",
    orig_text_name="en_norm",
    dest_text_name="ipa_norm",
    ignore_tones=True,
    ignore_arcs=True
  )

  preprocess_text(
    base_dir="/datasets/models/taco2pt_v5",
    ds_name="thchs",
    text_name="chn",
  )

  text_convert_to_ipa(
    base_dir="/datasets/models/taco2pt_v5",
    ds_name="thchs",
    orig_text_name="chn",
    dest_text_name="ipa",
    ignore_tones=False,
    ignore_arcs=True
  )
