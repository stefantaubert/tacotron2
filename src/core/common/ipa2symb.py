import string
from typing import List

from ipapy.ipastring import IPAString

from src.core.common.globals import PADDING_SYMBOL

# _rx = '[{}]'.format(re.escape(string.punctuation))

ARC = '͡'


def extract_from_sentence(ipa_sentence: str, ignore_tones: bool, ignore_arcs: bool):
  res = []
  tmp: List[str] = []

  for c in ipa_sentence:
    if c in string.punctuation or c in string.whitespace:
      if len(tmp) > 0:
        raw_word_symbols = _extract_symbols(tmp, ignore_tones, ignore_arcs)
        res.extend(raw_word_symbols)
        tmp.clear()
      res.append(c)
    else:
      tmp.append(c)

  if len(tmp) > 0:
    raw_word_symbols = _extract_symbols(tmp, ignore_tones, ignore_arcs)
    res.extend(raw_word_symbols)
    tmp.clear()
  return res


def _extract_symbols(input_symbols: List[str], ignore_tones: bool, ignore_arcs: bool, replace_unknown_ipa_by: str = PADDING_SYMBOL) -> List[str]:
  symbols: List[str] = []
  input_word = ''.join(input_symbols)
  try:
    ipa = IPAString(unicode_string=input_word, ignore=False)
  except:
    ipa = IPAString(unicode_string=input_word, ignore=True)
    print(f"{input_word} conversion to IPA failed. Result would be: {ipa}.")
    result = [replace_unknown_ipa_by] * len(input_symbols)
    return result

  for char in ipa.ipa_chars:
    if char.is_diacritic or char.is_tone:
      if len(symbols) > 0:
        if char.is_tone and ignore_tones:
          continue
        # I think it is a bug in IPAString that the arc sometimes gets classified as diacritic and sometimes not
        if char.unicode_repr == ARC:
          if ignore_arcs:
            continue
          symbols.append(ARC)
        else:
          symbols[-1] += char.unicode_repr
    else:
      uc = char.unicode_repr
      if ignore_arcs:
        uc = uc.split(ARC)
        symbols.extend(uc)
      else:
        symbols.append(uc)

  return symbols
