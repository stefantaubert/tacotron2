import re
from src.core.common.adjustments.whitespace import collapse_whitespace
from src.core.common.adjustments.abbreviations import expand_abbreviations
from src.core.common.adjustments.numbers import normalize_numbers
from dragonmapper import hanzi
from src.core.common.language import Language
from src.core.common.ipa2symb import extract_from_sentence
from collections import OrderedDict
from typing import List, OrderedDict as OrderedDictType, Set, Optional
from nltk.tokenize import sent_tokenize
from nltk import download
import epitran
from unidecode import unidecode as convert_to_ascii


_chn_question_mark = "？"
_chn_exklamation_mark = "！"
_chn_full_stop = "。"

_epitran_cache = {}


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


def en_to_ipa(text: str) -> str:
  if Language.ENG not in _epitran_cache.keys():
    _epitran_cache[Language.ENG] = epitran.Epitran('eng-Latn')
  result = _epitran_cache[Language.ENG].transliterate(text)
  return result


def ger_to_ipa(text: str) -> str:
  if Language.GER not in _epitran_cache.keys():
    _epitran_cache[Language.GER] = epitran.Epitran('deu-Latn')
  result = _epitran_cache[Language.GER].transliterate(text)
  return result


def normalize_en(text: str) -> str:
  text = convert_to_ascii(text)
  # text = text.lower()
  # todo datetime conversion, BC to beecee
  text = text.strip()
  text = normalize_numbers(text)
  text = expand_abbreviations(text)
  text = collapse_whitespace(text)
  return text


def normalize_ger(text: str) -> str:
  text = text.strip()
  text = collapse_whitespace(text)
  return text


def normalize_ipa(text: str) -> str:
  return text


def normalize_chn(text: str) -> str:
  text = text.strip()
  text = collapse_whitespace(text)
  return text


def normalize(text: str, lang: Language) -> str:
  if lang == Language.ENG:
    return normalize_en(text)
  elif lang == Language.GER:
    return normalize_ger(text)
  elif lang == Language.CHN:
    return normalize_chn(text)
  elif lang == Language.IPA:
    return normalize_ipa
  else:
    assert False


def convert_to_ipa(text: str, lang: Language) -> str:
  if lang == Language.ENG:
    return en_to_ipa(text)
  elif lang == Language.GER:
    return ger_to_ipa(text)
  elif lang == Language.CHN:
    return chn_to_ipa(text)
  elif lang == Language.IPA:
    return text
  else:
    assert False


def split_sentences(text: str, lang: Language) -> List[str]:
  if lang == Language.CHN:
    return split_chn_text(text)
  elif lang == Language.IPA:
    return split_ipa_text(text)
  elif lang == Language.ENG:
    return split_en_text(text)
  elif lang == Language.GER:
    return split_ger_text(text)
  else:
    assert False


def text_to_symbols(text: str, lang: Language, ignore_tones: Optional[bool] = None, ignore_arcs: Optional[bool] = None) -> List[str]:
  if lang == Language.ENG:
    symbols = list(text)
  elif lang == Language.GER:
    symbols = list(text)
  elif lang == Language.CHN:
    symbols = list(text)
  elif lang == Language.IPA:
    symbols = extract_from_sentence(
      text,
      ignore_tones=ignore_tones,
      ignore_arcs=ignore_arcs
    )
  else:
    assert False

  return symbols


def split_text(text: str, separators: List[str]) -> List[str]:
  pattern = "|".join(separators)
  sents = re.split(f'({pattern})', text)
  res = []
  for i, sent in enumerate(sents):
    if i % 2 == 0:
      res.append(sent)
      if i + 1 < len(sents):
        res[-1] += sents[i + 1]
  res = [x.strip() for x in res]
  res = [x for x in res if x]
  return res


def split_en_text(text: str) -> List[str]:
  download('punkt', quiet=True)
  res = sent_tokenize(text, language="english")
  return res


def split_ger_text(text: str) -> List[str]:
  download('punkt', quiet=True)
  res = sent_tokenize(text, language="german")
  return res


def split_ipa_text(text: str) -> List[str]:
  separators = [r"\?", r"\!", r"\."]
  return split_text(text, separators)


def split_chn_text(text: str) -> List[str]:
  separators = [r"\？", r"\！", r"\。"]
  return split_text(text, separators)


_chn_mappings = [
  (r"。", "."),
  (r"？", "?"),
  (r"！", "!"),
  (r"，", ","),
  (r"：", ":"),
  (r"；", ";"),
  (r"「", "\""),
  (r"」", "\""),
  (r"『", "\""),
  (r"』", "\""),
  (r"、", ",")
]

_subs = [(re.compile(regex_pattern), replace_with) for regex_pattern, replace_with in _chn_mappings]


def split_chn_sentence(sentence: str) -> List[str]:
  chn_words = sentence.split(' ')
  return chn_words


def chn_to_ipa(chn: str):
  chn_words = split_chn_sentence(chn)
  res = []
  for word in chn_words:
    chn_ipa = hanzi.to_ipa(word)
    chn_ipa = chn_ipa.replace(' ', '')
    res.append(chn_ipa)
  res_str = ' '.join(res)

  for regex, replacement in _subs:
    res_str = re.sub(regex, replacement, res_str)

  return res_str


if __name__ == "__main__":
  w = "东北军 的 一些 爱？」"
  a = chn_to_ipa(w)
  print(a)
