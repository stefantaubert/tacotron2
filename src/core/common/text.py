import re
from dragonmapper import hanzi
from src.core.common.language import Language
from src.core.common.ipa2symb import extract_from_sentence
from collections import OrderedDict
from typing import List, OrderedDict as OrderedDictType, Set
from nltk.tokenize import sent_tokenize
from nltk import download

_question_mark = "？"
_exklamation_mark = "！"
_full_stop = "。"

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
  res = OrderedDict([(s, i) for i, s in enumerate(unique_symbols)])
  return res

def switch_keys_with_values(dictionary: OrderedDictType) -> OrderedDictType:
  result = OrderedDict([(v, k) for k, v in dictionary.items()])
  return result

def split_sentences(text: str, lang: Language) -> List[str]:
  if lang == Language.ENG or lang == Language.GER:
    download('punkt', quiet=True)
  
  if lang == Language.CHN:
    sentences = split_chn_text(text)
  elif lang == Language.IPA:
    sentences = split_ipa_text(text)
  elif lang == Language.ENG:
    sentences = sent_tokenize(text, language="english")
  elif lang == Language.GER:
    sentences = sent_tokenize(text, language="german")
  else:
    raise Exception("Unknown input language!")

  return sentences

def split_text(text: str, separators: List[str]) -> List[str]:
  pattern = "|".join(separators)
  sents = re.split(f'({pattern})', text)
  res = []
  for i in range(len(sents)):
    if i % 2 == 0:
      res.append(sents[i])
      if i + 1 < len(sents):
        res[-1] += sents[i+1]
  res = [x.strip() for x in res]
  res = [x for x in res if x]
  return res

def split_ipa_text(text: str) -> List[str]:
  separators = ["?", "!", "."]
  return split_text(text, separators)

def split_chn_text(text: str) -> List[str]:
  separators = ["？", "！", "。"]
  return split_text(text, separators)
 
def extract_symbols(sentence: str, lang: Language):
  if lang == Language.ENG:
    symbols = list(sentence)
  elif lang == Language.GER:
    symbols = list(sentence)
  elif lang == Language.CHN:
    symbols = list(sentence)
  elif lang == Language.IPA:
    symbols = extract_from_sentence(sentence, ignore_tones=False, ignore_arcs=False, replace_unknown_ipa_by="")
  return symbols
  
_chn_mappings = [
  ("。", "."),
  ("？", "?"),
  ("！", "!"),
  ("，", ","),
  ("：", ":"),
  ("；", ";"),
  ("「", "\""),
  ("」", "\""),
  ("『", "\""),
  ("』", "\""),
  ("、", ",")
]

_subs = [(re.compile(f'\{x[0]}'), x[1]) for x in _chn_mappings]

def chn_to_ipa(chn: str):
  chn_words = chn.split(' ')
  res = []
  for w in chn_words:
    chn_ipa = hanzi.to_ipa(w)
    chn_ipa = chn_ipa.replace(' ', '')    
    res.append(chn_ipa)
  res_str = ' '.join(res)
  for regex, replacement in _subs:
    res_str = re.sub(regex, replacement, res_str)

  return res_str

if __name__ == "__main__":
  w = "东北军 的 一些 爱？」"
  res = chn_to_ipa(w)
  print(res)