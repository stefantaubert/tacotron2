import os

from text_utils import SymbolIdDict
from src.core.common.utils import get_subdir

_symbols_json = "symbols.json"


def get_pre_dir(base_dir: str, create: bool = False):
  return get_subdir(base_dir, 'pre', create)


def get_text_dir(prep_dir: str, text_name: str, create: bool):
  return get_subdir(prep_dir, text_name, create)


def load_text_symbol_converter(text_dir: str) -> SymbolIdDict:
  path = os.path.join(text_dir, _symbols_json)
  return SymbolIdDict.load_from_file(path)


def save_text_symbol_converter(text_dir: str, data: SymbolIdDict):
  path = os.path.join(text_dir, _symbols_json)
  data.save(path)
