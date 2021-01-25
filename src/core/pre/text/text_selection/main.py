import string
from collections import Counter
from logging import StringTemplateStyle
from typing import Dict, List, Set, Tuple, Type, TypeVar, Union

from src.app.pre.ds import get_ds_dir
from src.app.pre.text import (get_text_dir, load_text_csv,
                              load_text_symbol_converter)
from src.core.pre.text.pre import TextData
from src.core.pre.text.text_selection.greedy import set_cover, set_cover_dict
from text_utils import get_ngrams

x = {1: "a"}
b = {2: "c"}
x.update(b)
print(x)


base_dir = "/home/mi/data/tacotron2"
ds_dir = get_ds_dir(base_dir, ds_name="ljs")
text_dir = get_text_dir(ds_dir, text_name="ipa_norm_epi")
text = load_text_csv(text_dir)
orig_conv = load_text_symbol_converter(text_dir)


res = set_cover_dict(
  subsets={
    1: set(["a", "b"]),
    2: set(["a", "c"]),
    3: set(["a", "c", "e"]),
    4: set(["a", "c", "e", "f"]),
  }
)

print(res)

res = set_cover(
  subsets=[set(["a", "b"]), set(["b", "a"]), set(["b", "a"]), set(["c", "b"])]
)
print(res)

all_two_ngrams_list: List[Tuple] = []
all_two_ngrams_dict: Dict[int, List[Tuple]] = {}

for item in text.items(True):
  symbols = orig_conv.get_symbols(item.serialized_symbol_ids)
  two_ngram = get_ngrams(symbols, 2)
  all_two_ngrams_list.append(two_ngram)
  all_two_ngrams_dict[item.entry_id] = two_ngram

all_two_ngrams = set(y for x in all_two_ngrams_list for y in x)

all_two_ngrams_dict_set: Dict[int, List[Tuple]] = {}
all_two_ngrams_dict_set = {k: set(v) for k, v in all_two_ngrams_dict.items()}
res = set_cover_dict(all_two_ngrams_dict_set)
print(res.keys())

for x in text.items():
  if x.entry_id in res:
    print(x.text)
print(len(res))

all_two_ngrams_set = [set(x) for x in all_two_ngrams_list]
res = set_cover(all_two_ngrams_set)
print(len(res))
print(res)
