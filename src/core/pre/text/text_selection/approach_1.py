from math import ceil
from typing import Dict, List, Set, Tuple, Type, TypeVar, Union

from matplotlib.pyplot import savefig
from src.core.pre.text.pre import TextDataList
from src.core.pre.text.text_selection.greedy import (set_cover_dict,
                                                     set_cover_dict_max_length,
                                                     set_cover_n_chars)
from src.core.pre.text.text_selection.helper import contains_all
from text_utils.text import get_ngrams

_T = TypeVar('_T')


def get_min_shards(text_data: Dict[int, List[str]], shards: int) -> int:
  lengths = {i: len(symbols) for i, symbols in text_data.items()}
  symbols_count: int = len(set(y for x in text_data.values() for y in x))
  disymbols_dict: Dict[int, List[Tuple[str, str]]] = {
    i: get_ngrams(symbols, 2) for i, symbols in text_data.items()}
  disymbols_dict_set: Dict[int, Set[Tuple]] = {k: set(v) for k, v in disymbols_dict.items()}

  res = set_cover_dict(disymbols_dict_set)
  min_chars_to_cover_all_diphones = sum([lengths[x] for x in res.keys()])
  min_shards = min_chars_to_cover_all_diphones / symbols_count
  min_shards = ceil(min_shards)


def main(text_data: Dict[int, List[str]], shards: int) -> Dict[int, List[str]]:
  lengths = {i: len(symbols) for i, symbols in text_data.items()}
  all_symbols: Set[str] = set(y for x in text_data.values() for y in x)
  symbols_count: int = len(all_symbols)
  disymbols_dict: Dict[int, List[Tuple[str, str]]] = {
    i: get_ngrams(symbols, 2) for i, symbols in text_data.items()}
  disymbols: Set[Tuple[str, str]] = set(y for x in disymbols_dict.values() for y in x)
  disymbols_dict_set: Dict[int, Set[Tuple]] = {k: set(v) for k, v in disymbols_dict.items()}
  max_chars = symbols_count * shards
  print(f"Using {shards} shards for {symbols_count} symbols, resulting in {max_chars} chars.")
  result, rest, final_count = set_cover_n_chars(disymbols_dict_set, lengths, max_chars)
  contains_all_disymbols, not_included_disymbols = contains_all(result.values(), disymbols)
  res = {k: text_data[k] for k in result}
  contains_all_symbols, not_included_symbols = contains_all(res.values(), all_symbols)
  print(f"Result contain all disymbols: {contains_all_disymbols}")
  print(f"Result contain all symbols: {contains_all_symbols}")
  if not contains_all_disymbols:
    print("Not included disymbols are:")
    print(not_included_disymbols)
    print(f"Count: {len(not_included_disymbols)}/{len(disymbols)} ({len(not_included_disymbols) / len(disymbols) * 100:.2f}%)")
  if not contains_all_symbols:
    print("Not included symbols are:")
    print(not_included_symbols)
    print(f"Count: {len(not_included_symbols)}/{len(all_symbols)} ({len(not_included_symbols) / len(all_symbols) * 100:.2f}%)")

  return res


if __name__ == "__main__":
  from src.app.pre.ds import get_ds_dir
  from src.app.pre.text import (get_text_dir, load_text_csv,
                                load_text_symbol_converter)
  from src.app.pre.wav import get_wav_dir, load_wav_csv

  base_dir = "/home/mi/data/tacotron2"
  ds_dir = get_ds_dir(base_dir, ds_name="ljs")
  text_dir = get_text_dir(ds_dir, text_name="ipa_norm_both")
  wav_dir = get_wav_dir(ds_dir, "22050Hz")
  text = load_text_csv(text_dir)
  orig_conv = load_text_symbol_converter(text_dir)
  text_data = {x.entry_id: orig_conv.get_symbols(x.serialized_symbol_ids) for x in text.items()}
  res = main(
    text_data=text_data,
    shards=2000,
  )
