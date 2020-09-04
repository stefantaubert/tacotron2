from src.core.common import convert_to_ipa, normalize
from src.core.common import text_to_symbols
from src.core.common import SymbolIdDict
from src.core.common import Language
from typing import List, Tuple


def symbols_normalize(symbols: List[str], lang: Language, accent_ids: List[str]) -> Tuple[List[str], List[int]]:
  assert len(symbols) == len(accent_ids)
  orig_text = SymbolIdDict.symbols_to_str(symbols)
  text = normalize(orig_text, lang)
  new_symbols: List[str] = text_to_symbols(text, lang)
  if lang != Language.IPA:
    if len(accent_ids) > 0:
      new_accent_ids = [accent_ids[0]] * len(symbols)
    else:
      new_accent_ids = []
  else:
    # because no replacing was done in ipa normalization
    # maybe support remove whitespace
    new_accent_ids = accent_ids
  assert len(new_symbols) == len(new_accent_ids)
  return new_symbols, new_accent_ids


def symbols_convert_to_ipa(symbols: List[str], lang: Language, accent_ids: List[str], ignore_tones: bool, ignore_arcs: bool, replace_unknown_ipa_by: str) -> Tuple[List[str], List[int]]:
  assert len(symbols) == len(accent_ids)
  if lang != Language.IPA:
    orig_text = SymbolIdDict.symbols_to_str(symbols)
    ipa = convert_to_ipa(orig_text, lang)
    new_symbols: List[str] = text_to_symbols(
      ipa, Language.IPA, ignore_tones, ignore_arcs, replace_unknown_ipa_by=replace_unknown_ipa_by)
    if len(accent_ids) > 0:
      new_accent_ids = [accent_ids[0]] * len(new_symbols)
    else:
      new_accent_ids = []
    assert len(new_symbols) == len(new_accent_ids)
    return new_symbols, new_accent_ids
  else:
    return symbols, accent_ids
