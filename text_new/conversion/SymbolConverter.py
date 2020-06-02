import json

import numpy as np

# padding
_pad = '_'

# end of string
_eos = '~'

class SymbolConverter():
  '''
  Defines the set of symbols used in text input to the model.
  '''

  def _init_from_symbols(self, symbols: set):
    all_symbols = list(_pad) + list(_eos) + list(symbols)
    sorted_symbols = sorted(all_symbols)
    # set() because if pad and eos in symbols they were ignored
    self._id_to_symbol = {s: i for i, s in enumerate(sorted_symbols)}
    self._init_properties()
  
  def get_symbols(self) -> set:
    result = set(self._id_to_symbol.keys())
    return result

  def get_symbols_count(self) -> int:
    result = len(self._id_to_symbol)
    return result

  def _init_properties(self):
    self._symbol_to_id = {v: k for k, v in self._id_to_symbol.items()}
    self._pad_id = self._id_to_symbol[_pad]
    self._eos_id = self._id_to_symbol[_eos]

  def _init_from_file(self, from_file: str):
    self._parse_from_file(from_file)

  def _parse_from_file(self, file_path: str):
    with open(file_path, 'r', encoding='utf-8') as f:
      self._id_to_symbol = json.load(f)

    self._init_properties()

  def dump(self, file_path: str):
    with open(file_path, 'w', encoding='utf-8') as f:
      json.dump(self._id_to_symbol, f)

  def remove_unknown_symbols(self, symbols):
    result = [symbol for symbol in symbols if symbol in self._id_to_symbol.keys()]
    return result
    
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

  def sequence_to_original_text(self, sequence):
    '''Converts a sequence of IDs back to a string'''
    symbols = [self._get_symbol(s_id) for s_id in sequence if self._is_valid_text_symbol(self._get_symbol(s_id))]
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
    return symbol is not _pad and symbol is not _eos

  def _get_valid_symbolids(self, symbols):
    res = []
    for symbol in symbols:
      if self._is_valid_text_symbol(symbol):
        s_id = self._get_id(symbol)
        res.append(s_id)
      else:
        print("Unknown symbol:", symbol)
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
  x = converter.get_symbols()
  filepath = '/datasets/tmp/dump.json'
  converter.dump(filepath)
  res = get_from_file(filepath)
  x = res.get_symbols()
  print(res.get_symbols_count())

  inp = "hello my name is mr. test"
  outp = converter.text_to_sequence(inp)
  print(outp.get_symbols_count())
  print(outp)

  outp_to_text = converter.sequence_to_text(outp)
  print(outp_to_text)
