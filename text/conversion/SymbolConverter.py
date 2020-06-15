import json

import numpy as np

# padding
_pad = '_'

# end of string
_eos = '~'

_initial_subset_id = 0

def get_symbols_from_str(symbols_str: str):
  sentences_symbols = symbols_str.split(',')
  sentences_symbols = list(map(int, sentences_symbols))
  return sentences_symbols
  
class SymbolConverter():
  '''
  Defines the set of symbols used in text input to the model.
  '''

  def _init_from_symbols(self, symbols: set):
    has_no_double_symbols = len(set(symbols)) == len(symbols)
    assert has_no_double_symbols
    all_symbols = list(_pad) + list(_eos) + list(symbols)
    sorted_symbols = sorted(all_symbols)
    # set() because if pad and eos in symbols they were ignored
    self._id_symbol_dict = {i: (_initial_subset_id, s) for i, s in enumerate(sorted_symbols)}
    self.actualize_symbol_id_dict()
    self._pad_id = self._get_id_from_symbol(_pad)
    self._eos_id = self._get_id_from_symbol(_eos)

  def actualize_symbol_id_dict(self):
    tmp = {}
    for s_id, data in self._id_symbol_dict.items():
      subset, symbol = data
      if symbol not in tmp.keys():
        tmp[symbol] = {}
      tmp[symbol][subset] = s_id

    self._symbol_id_dict = tmp

  def _get_symbol_from_id(self, symbol_id: int):
    subset, symbol = self._id_symbol_dict[symbol_id]
    return symbol

  def _get_id_from_symbol(self, symbol: str, subset_id_if_multiple: int = _initial_subset_id):
    subsets = self._symbol_id_dict[symbol]
    symbol_has_multiple_ids = len(subsets.keys()) > 1
    if symbol_has_multiple_ids:
      return self._symbol_id_dict[symbol][subset_id_if_multiple]
    else:
      return list(self._symbol_id_dict[symbol].values())[0]

  def get_symbols(self) -> set:
    result = set(self._symbol_id_dict.keys())
    return result

  def get_symbol_ids_count(self) -> int:
    result = len(self._id_symbol_dict)
    return result

  def _init_from_file(self, from_file: str):
    self._parse_from_file(from_file)

  def _parse_from_file(self, file_path: str):
    with open(file_path, 'r', encoding='utf-8') as f:
      self._id_symbol_dict = json.load(f)

    self.actualize_symbol_id_dict()
    self._pad_id = self._get_id_from_symbol(_pad)
    self._eos_id = self._get_id_from_symbol(_eos)

  def dump(self, file_path: str):
    with open(file_path, 'w', encoding='utf-8') as f:
      json.dump(self._id_symbol_dict, f)

  def plot(self, file_path: str, sort=True):
    with open(file_path, 'w', encoding='utf-8') as f:
      symbols = []
      for _, data in self._id_symbol_dict.items():
        subset_id, symbol = data
        symbols.append('{}{}'.format(subset_id, symbol))
        
      if sort:
        symbols = list(sorted(symbols))
      res = '\n'.join(symbols)
      f.write(res)
    
  def remove_unknown_symbols(self, symbols):
    result = [symbol for symbol in symbols if symbol in self._symbol_id_dict.keys()]
    return result

  def add_symbols(self, symbols: set, ignore_existing: bool, subset_id: int):
    #contains_already_existing_symbols = len(set(self._symbol_id_dict.keys()).intersection(symbols)) > 0
    #assert not contains_already_existing_symbols
    max_number = max(self._id_symbol_dict.keys())
    for i, new_symbol in enumerate(symbols):
      new_id = max_number + 1 + i
      if new_symbol in self._symbol_id_dict.keys():
        if ignore_existing:
          continue
      self._id_symbol_dict[new_id] = (subset_id, new_symbol)

    self.actualize_symbol_id_dict()

  def get_unknown_symbols(self, chars):
    unknown_symbols = set([x for x in chars if not self._is_valid_text_symbol(x)])
    return unknown_symbols

  # todo rename to symbols_to_sequence
  def text_to_sequence(self, chars):
    '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

      Args:
        text: string to convert to a sequence

      Returns:
        List of integers corresponding to the symbols in the text
    '''
    sequence = []

    ids = self._get_valid_symbolids(chars)
    sequence.extend(ids)

    # Append EOS token
    sequence.append(self._eos_id)

    result = np.asarray(sequence, dtype=np.int32)

    return result

  def sequence_to_text(self, sequence):
    '''Converts a sequence of IDs back to a string'''
    symbols = [self._get_symbol(s_id) for s_id in sequence]
    result = ''.join(symbols)

    return result

  def sequence_to_original_chars(self, sequence):
    '''Converts a sequence of IDs back to a string'''
    symbols = [self._get_symbol(s_id) for s_id in sequence if self._is_valid_text_symbol(self._get_symbol(s_id))]

    return symbols

  def sequence_to_original_text(self, sequence):
    '''Converts a sequence of IDs back to a string'''
    symbols = self.sequence_to_original_chars(sequence)
    result = ''.join(symbols)

    return result

  def _get_id(self, symbol: str) -> int:
    assert symbol in self._id_to_symbol
    result = self._id_to_symbol[symbol]
    return result

  def _get_symbol(self, symbol_id: int) -> str:
    assert symbol_id in self._symbol_to_id
    result = self._symbol_to_id[symbol_id]
    return result

  def _is_valid_text_id(self, text_id: int) -> bool:
    return text_id is not self._pad_id and text_id is not self._eos_id

  def _is_valid_text_symbol(self, symbol: str) -> bool:
    is_valid = symbol in self._id_to_symbol and symbol is not _pad and symbol is not _eos
    if not is_valid:
      x = 1
    return is_valid

  def _get_valid_symbolids(self, symbols):
    res = []
    for symbol in symbols:
      if self._is_valid_text_symbol(symbol):
        s_id = self._get_id(symbol)
        res.append(s_id)
      #else:
        #print("Unknown symbol:", symbol)
    return res

def get_from_symbols(symbols: set) -> SymbolConverter:
  instance = SymbolConverter()
  instance._init_from_symbols(symbols)
  return instance

def get_from_file(filepath: str) -> SymbolConverter:
  instance = SymbolConverter()
  instance._init_from_file(filepath)
  return instance


if __name__ == "__main__":
  symbols = set(['a', 'b', 'c', '!'])
  converter = get_from_symbols(symbols)
  symbols2 = ['a', 'x']
  converter.add_symbols(symbols2, ignore_existing=False, subset_id=2)

  x = converter.get_symbols()
  filepath = '/tmp/dump.json'
  converter.dump(filepath)
  converter.plot('/tmp/dump.txt')
  res = get_from_file(filepath)
  x = res.get_symbols()
  print(x)
  print(res.get_symbol_ids_count())

  inp = "hello my name is mr. test"
  outp = converter.text_to_sequence(inp)
  print(outp.get_symbol_ids_count())
  print(outp)

  outp_to_text = converter.sequence_to_text(outp)
  print(outp_to_text)
