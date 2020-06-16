import json

import numpy as np

from collections import OrderedDict 

def deserialize_symbol_ids(searialized_str: str):
  sentences_symbols = searialized_str.split(',')
  sentences_symbols = list(map(int, sentences_symbols))
  return sentences_symbols

def serialize_symbol_ids(symbol_ids: list):
  sentences_symbols = ','.join(symbol_ids)
  return sentences_symbols

# padding
_pad = '_'

# end of string
_eos = '~'

_initial_subset_id = 0

def _get_id_symbol_dict(symbols: list) -> OrderedDict:
  has_no_double_symbols = len(set(symbols)) == len(symbols)
  assert has_no_double_symbols
  res = OrderedDict([(i, (_initial_subset_id, s)) for i, s in enumerate(symbols)])
  return res

def _get_symbol_id_dict(id_symbol_dict: OrderedDict) -> OrderedDict:
  tmp = OrderedDict()
  for s_id, data in id_symbol_dict.items():
    subset, symbol = data
    if symbol not in tmp.keys():
      tmp[symbol] = {}
    tmp[symbol][subset] = s_id
  return tmp

def _symbols_to_dict(symbols: set) -> OrderedDict:
  has_no_double_symbols = len(set(symbols)) == len(symbols)
  assert has_no_double_symbols
  if _pad in symbols:
    symbols.remove(_pad)
  if _eos in symbols:
    symbols.remove(_eos)
  all_symbols = list(_pad) + list(_eos) + list(sorted(set(symbols)))
  # set() because if pad and eos in symbols they were ignored
  id_symbol_dict = _get_id_symbol_dict(all_symbols)
  return id_symbol_dict

def _file_to_dict(filepath: str) -> OrderedDict:
  with open(filepath, 'r', encoding='utf-8') as f:
    id_symbol_dict = json.load(f)
  return id_symbol_dict

class SymbolConverter():

  '''
  Defines the set of symbols used in text input to the model.
  '''
  def __init__(self, id_symbol_dict: OrderedDict):
    super().__init__()
    self._id_symbol_dict = id_symbol_dict
    self._symbol_id_dict = _get_symbol_id_dict(id_symbol_dict)
  
  def id_to_symbol(self, symbol_id: int):
    assert symbol_id in self._id_symbol_dict.keys()
    subset, symbol = self._id_symbol_dict[symbol_id]
    return symbol

  def symbol_to_id(self, symbol: str, subset_id_if_multiple: int = _initial_subset_id):
    subsets = self._symbol_id_dict[symbol]
    symbol_has_multiple_ids = len(subsets.keys()) > 1
    if symbol_has_multiple_ids:
      assert subset_id_if_multiple in self._symbol_id_dict[symbol].keys()
      return self._symbol_id_dict[symbol][subset_id_if_multiple]
    else:
      return list(self._symbol_id_dict[symbol].values())[0]

  def get_symbols(self, include_subset_id: bool = True, include_id: bool = True) -> list:
    symbols = []
    for symbol_id, data in self._id_symbol_dict.items():
      subset_id, symbol = data
      tmp = ''
      if include_subset_id:
        tmp += '{}\t'.format(subset_id)
      tmp += '{}'.format(symbol)
      if include_id:
        tmp += '\t{}'.format(symbol_id)
      symbols.append(tmp)
    return symbols

  def get_symbol_ids_count(self) -> int:
    result = len(self._id_symbol_dict)
    return result

  def dump(self, file_path: str):
    with open(file_path, 'w', encoding='utf-8') as f:
      json.dump(self._id_symbol_dict, f)

  def plot(self, file_path: str, sort=True):
    with open(file_path, 'w', encoding='utf-8') as f:
      symbols = self.get_symbols(include_id=True, include_subset_id=True)
      if sort:
        symbols = list(sorted(symbols))
      res = '\n'.join(symbols)
      f.write(res)
    
  def __replace_unknown_symbols(self, symbols: list, replace_with_symbol: str = None) -> list:
    assert replace_with_symbol == None or replace_with_symbol in self._symbol_id_dict.keys()
    result = []
    for symbol in symbols:
      if symbol in self._symbol_id_dict.keys():
        result.append(symbol)
      elif replace_with_symbol != None:
        result.append(replace_with_symbol)
    return result

  def add_symbols(self, symbols: set, ignore_existing: bool, subset_id: int):
    #contains_already_existing_symbols = len(set(self._symbol_id_dict.keys()).intersection(symbols)) > 0
    #assert not contains_already_existing_symbols
    sorted_symbols = list(sorted(symbols))
    next_id = self.get_symbol_ids_count()
    for i, new_symbol in enumerate(sorted_symbols):
      symbol_already_exists_and_should_be_ignored = new_symbol in self._symbol_id_dict.keys() and ignore_existing
      if symbol_already_exists_and_should_be_ignored:
        continue
      self._id_symbol_dict[next_id] = (subset_id, new_symbol)
      next_id += 1

    self._symbol_id_dict = _get_symbol_id_dict(self._id_symbol_dict)

  def get_unknown_symbols(self, symbols: list):
    unknown_symbols = set([x for x in symbols if not self.__is_text_symbol(x)])
    return unknown_symbols

  def symbols_to_ids(self, symbols: list, subset_id_if_multiple: int = _initial_subset_id, add_eos: bool = True, replace_unknown_with_pad: bool = True) -> np.ndarray:
    valid_symbols = self.__replace_unknown_symbols(symbols, _pad if replace_unknown_with_pad else None)
    # Append EOS token
    if add_eos:
      valid_symbols.append(_eos)
    ids = [self.symbol_to_id(x, subset_id_if_multiple) for x in valid_symbols]
    result = np.asarray(ids, dtype=np.int32)
    return result

  def ids_to_symbols(self, symbol_ids: list) -> list:
    symbols = [self.id_to_symbol(s_id) for s_id in symbol_ids]
    return symbols

  def ids_to_text(self, symbol_ids: list) -> str:
    symbols = self.ids_to_symbols(symbol_ids)
    text = ''.join(symbols)
    return text

def load_from_file_v1(filepath: str) -> SymbolConverter:
  old = _file_to_dict(filepath)
  all_symbols = list(old.keys())
  # take index as id is safe
  new_dict = _get_id_symbol_dict(all_symbols)
  return SymbolConverter(new_dict)

def load_from_file_v2(filepath: str) -> SymbolConverter:
  return SymbolConverter(_file_to_dict(filepath))

def load_from_file(filepath: str) -> SymbolConverter:
  d = _file_to_dict(filepath)
  values = list(d.values())
  assert len(values) > 0
  dtype_of_values_is_list = type(values[0]) is list
  if dtype_of_values_is_list:
    return load_from_file_v2(filepath)
  else:
    return load_from_file_v1(filepath)

  return SymbolConverter(_file_to_dict(filepath))

def init_from_symbols(symbols: set) -> SymbolConverter:
  return SymbolConverter(_symbols_to_dict(symbols))
