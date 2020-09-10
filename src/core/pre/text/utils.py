from typing import List, Tuple

from src.core.common.language import Language
from src.core.common.symbol_id_dict import SymbolIdDict
from src.core.common.text import convert_to_ipa, normalize, text_to_symbols


def symbols_normalize(symbols: List[str], lang: Language, accent_ids: List[str]) -> Tuple[List[str], List[int]]:
  assert len(symbols) == len(accent_ids)
  orig_text = SymbolIdDict.symbols_to_str(symbols)
  text = normalize(orig_text, lang)
  new_symbols: List[str] = text_to_symbols(text, lang)
  if lang != Language.IPA:
    if len(accent_ids) > 0:
      new_accent_ids = [accent_ids[0]] * len(new_symbols)
    else:
      new_accent_ids = []
  else:
    # because no replacing was done in ipa normalization
    # maybe support remove whitespace
    new_accent_ids = accent_ids
  assert len(new_symbols) == len(new_accent_ids)
  return new_symbols, new_accent_ids


def symbols_convert_to_ipa(symbols: List[str], lang: Language, accent_ids: List[str], ignore_tones: bool, ignore_arcs: bool) -> Tuple[List[str], List[int]]:
  assert len(symbols) == len(accent_ids)
  # Note: do also for ipa symbols to have possibility to remove arcs and tones
  orig_text = SymbolIdDict.symbols_to_str(symbols)
  ipa = convert_to_ipa(orig_text, lang)
  new_symbols: List[str] = text_to_symbols(ipa, Language.IPA, ignore_tones, ignore_arcs)
  if len(accent_ids) > 0:
    new_accent_ids = [accent_ids[0]] * len(new_symbols)
  else:
    new_accent_ids = []
  assert len(new_symbols) == len(new_accent_ids)
  return new_symbols, new_accent_ids
