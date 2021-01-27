import random
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


def main(text_data: Dict[int, List[str]], shards: int) -> Dict[int, List[str]]:
  lengths = {i: len(symbols) for i, symbols in text_data.items()}
  all_symbols: Set[str] = set(y for x in text_data.values() for y in x)
  symbols_count: int = len(all_symbols)

  max_chars = symbols_count * shards

  res_keys = rand_n_chars(lengths, max_chars, seed=4242)
  res: Dict[int, List[str]] = {k: text_data[k] for k in res_keys}
  contains_all_symbols, not_included_symbols = contains_all(res.values(), all_symbols)
  print(f"Result contains all symbols: {contains_all_symbols}")
  if not contains_all_symbols:
    print("Not included are:")
    print(not_included_symbols)
    print(f"Count: {len(not_included_symbols)}/{len(all_symbols)} ({len(not_included_symbols) / len(all_symbols) * 100:.2f}%)")

  disymbols_dict: Dict[int, List[Tuple[str, str]]] = {
    i: get_ngrams(symbols, 2) for i, symbols in text_data.items()}
  disymbols: Set[Tuple[str, str]] = set(y for x in disymbols_dict.values() for y in x)
  res_disymbols: Dict[int, List[str]] = {k: disymbols_dict[k] for k in res_keys}
  contains_all_disymbols, not_included_disymbols = contains_all(res_disymbols.values(), disymbols)
  print(f"Result contains all disymbols: {contains_all_disymbols}")
  if not contains_all_disymbols:
    print("Not included disymbols are:")
    print(not_included_disymbols)
    print(f"Count: {len(not_included_disymbols)}/{len(disymbols)} ({len(not_included_disymbols) / len(disymbols) * 100:.2f}%)")
  return res


def rand_n_chars(lengths: Dict[int, int], n_max: int, seed: int) -> Set[int]:
  result: Set[int] = set()
  rest_lenghts = {k: v for k, v in lengths.items()}
  random.seed(seed)
  total_length = 0
  while total_length <= n_max:
    next_key = random.choice(list(rest_lenghts.keys()))
    length = rest_lenghts[next_key]
    new_length = total_length + length
    if new_length <= n_max:
      total_length = new_length
      result.add(next_key)
      rest_lenghts.pop(next_key)
    else:
      break
  print(
    f"Extracted {total_length} chars from {len(result)} out of {len(lengths)} utterances ({len(lengths) - len(result)} remain).")
  return result


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
