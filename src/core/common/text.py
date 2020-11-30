from collections import OrderedDict
from typing import List
from typing import OrderedDict as OrderedDictType
from typing import Set


def deserialize_list(serialized_str: str) -> List[int]:
  sentences_symbols = serialized_str.split(',')
  sentences_symbols = list(map(int, sentences_symbols))
  return sentences_symbols


def serialize_list(symbol_ids: List[int]) -> str:
  sentences_symbols = list(map(str, symbol_ids))
  sentences_symbols = ','.join(sentences_symbols)
  return sentences_symbols


def get_entries_ids_dict(symbols: Set[str]) -> OrderedDictType[str, int]:
  unique_symbols = list(sorted(set(symbols)))
  return get_entries_ids_dict_order(unique_symbols)


def get_entries_ids_dict_order(symbols: List[str]) -> OrderedDictType[str, int]:
  assert len(symbols) == len(set(symbols))
  res = OrderedDict([(s, i) for i, s in enumerate(symbols)])
  return res


def switch_keys_with_values(dictionary: OrderedDictType) -> OrderedDictType:
  result = OrderedDict([(v, k) for k, v in dictionary.items()])
  return result
