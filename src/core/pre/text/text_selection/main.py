import pickle
import string
from collections import Counter
from logging import StringTemplateStyle
from math import ceil
from typing import Dict, List, Set, Tuple, Type, TypeVar, Union

import matplotlib.pyplot as plt
import numpy as np
from src.app.pre.ds import get_ds_dir
from src.app.pre.text import (get_text_dir, load_text_csv,
                              load_text_symbol_converter)
from src.app.pre.wav import get_wav_dir, load_wav_csv
from src.core.common.utils import contains_only_allowed_symbols
from src.core.pre.text.pre import TextData
from src.core.pre.text.text_selection.greedy import (set_cover, set_cover_dict,
                                                     set_cover_dict_max_length,
                                                     set_cover_n)
from text_utils import get_ngrams


def save_obj(obj, name):
  with open('/tmp/' + name + '.pkl', 'wb') as f:
    pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name) -> Dict[str, List]:
  with open('/tmp/' + name + '.pkl', 'rb') as f:
    return pickle.load(f)


allowed_symbols = set([" ", "ˈ", "ʌ", "ɪ", "t", "n", "d", "s", "ɹ", "l", "ð", "i", "k", "z", "ɛ", "m", "ɝ", "æ", "w", "p",
                       "v", "ɑ", "f", "ʊ", "ɔ", "a", "b", "u", ",", "e", "h", "ʃ", "o", "ŋ", "ɡ", "ʒ", "j", "θ", "ə", "ɹ̩", "?", "!", ".", "ˌ"])

# x = contains_only_allowed_symbols(["t", "x"], allowed_symbols)

removed = "-", "\"", ";", ":", "(", ")", "'", "[", "]"

res = load_obj("set_cover_1")

all_two_grams = [x for y in res.values() for x in y]
old_len = len(all_two_grams)
all_two_grams = [x for x in all_two_grams if contains_only_allowed_symbols(x, allowed_symbols)]

print(f"Removed {old_len - len(all_two_grams)} from {old_len} ({100- (len(all_two_grams) / old_len * 100):.2f}%) diphones with not allowed symbols.")

c_ngrams_two = Counter(all_two_grams)

c_occurences = Counter(c_ngrams_two.values())

diphone_count = c_occurences.keys()
diphone_count_occurences = c_occurences.values()

fig, ax = plt.subplots(figsize=(6, 6))
ax.bar(diphone_count, diphone_count_occurences)
ax.set_title('There are y diphones that occur x-times.')
ax.set_ylabel('y')
ax.set_xlabel('x')
fig.savefig("playground.pdf")


diphone_count = {1: "a"}
b = {2: "c"}
diphone_count.update(b)
print(diphone_count)


base_dir = "/home/mi/data/tacotron2"
ds_dir = get_ds_dir(base_dir, ds_name="ljs")
text_dir = get_text_dir(ds_dir, text_name="ipa_norm_both")
wav_dir = get_wav_dir(ds_dir, "22050Hz")
text = load_text_csv(text_dir)
wav = load_wav_csv(wav_dir)
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
all_symbols: Set[str] = set()
ignored = 0
for item in text.items(True):
  symbols = orig_conv.get_symbols(item.serialized_symbol_ids)
  if contains_only_allowed_symbols(symbols, allowed_symbols):
    all_symbols |= set(symbols)
    two_ngram = get_ngrams(symbols, 2)
    all_two_ngrams_list.append(two_ngram)
    all_two_ngrams_dict[item.entry_id] = two_ngram
  else:
    #print(f"Ignored: {item.text}")
    ignored += 1
print(f"Removed {ignored} from {len(text)} ({ignored / len(text) * 100:.2f}%) diphones with not allowed symbols.")
all_two_ngrams = set(y for x in all_two_ngrams_list for y in x)

all_two_ngrams_dict_set: Dict[int, List[Tuple]] = {}
all_two_ngrams_dict_set = {k: set(v) for k, v in all_two_ngrams_dict.items()}
lengths = {i.entry_id: len(i.text) for i in text.items()}
all_symbols_count = len(all_symbols)
shards = 700
res = set_cover_dict(all_two_ngrams_dict_set)
min_chars_to_cover_all_diphones = sum([lengths[x] for x in res.keys()])
min_shards = min_chars_to_cover_all_diphones / all_symbols_count
min_shards = ceil(min_shards)
max_chars = all_symbols_count * shards
max_chars = all_symbols_count * min_shards
finished, res = set_cover_dict_max_length(all_two_ngrams_dict_set, lengths, max_chars)
#res = set_cover_n(all_two_ngrams_dict_set, 1)
print(f"Finished: {finished}")
save_obj(res, "set_cover_1")

# 1min 1000 chars ljs

print(res.keys())

total_duration = 0
total_charcount = 0
for t, w in zip(text.items(), wav.items()):
  if t.entry_id in res:
    # print(t.text)
    total_charcount += len(t.text)
    total_duration += w.duration
print(f"# {len(res)}")
print(f"Total duration: {total_duration / 60:.2f}min")
print(f"Total chars: {total_charcount}")

all_two_ngrams_set = [set(x) for x in all_two_ngrams_list]
res = set_cover(all_two_ngrams_set)
print(len(res))
print(res)
