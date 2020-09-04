from collections import Counter, OrderedDict
from dataclasses import dataclass
from src.core.pre.text.utils import symbols_convert_to_ipa, symbols_normalize
from src.core.common.utils import GenericList
from typing import List
from typing import OrderedDict as OrderedDictType
from typing import Tuple

from tqdm import tqdm

from src.core.common import parse_json, save_json, Language, SymbolIdDict, SymbolsDict, AccentsDict, deserialize_list, serialize_list, text_to_symbols, convert_to_ipa as text_convert_to_ipa, normalize as text_normalize, get_unique_items, get_counter
from src.core.pre.ds import DsData, DsDataList


@dataclass()
class TextData:
  entry_id: int
  text: str
  serialized_symbol_ids: str
  serialized_accent_ids: str
  lang: Language


class TextDataList(GenericList[TextData]):
  pass


def convert_to_ipa(data: TextDataList, symbol_converter: SymbolIdDict, ignore_tones: bool, ignore_arcs: bool, replace_unknown_ipa_by: str) -> Tuple[TextDataList, SymbolIdDict, SymbolsDict]:
  processed_data: List[Tuple[int, List[str], List[int], Language]] = []

  values: TextData
  for values in tqdm(data.items()):
    new_symbols, new_accent_ids = symbols_convert_to_ipa(
      symbols=symbol_converter.get_symbols(values.serialized_symbol_ids),
      lang=values.lang,
      accent_ids=deserialize_list(values.serialized_accent_ids),
      ignore_arcs=ignore_arcs,
      ignore_tones=ignore_tones,
      replace_unknown_ipa_by=replace_unknown_ipa_by
    )
    processed_data.append((values.entry_id, new_symbols, new_accent_ids, Language.IPA))

  return _prepare_data(processed_data)


def normalize(data: TextDataList, symbol_converter: SymbolIdDict) -> Tuple[TextDataList, SymbolIdDict, SymbolsDict]:
  processed_data: List[Tuple[int, List[str], List[int], Language]] = []

  values: TextData
  for values in tqdm(data):
    new_symbols, new_accent_ids = symbols_normalize(
      symbols=symbol_converter.get_symbols(values.serialized_symbol_ids),
      lang=values.lang,
      accent_ids=deserialize_list(values.serialized_accent_ids),
    )

    processed_data.append((values.entry_id, new_symbols, new_accent_ids, values.lang))

  return _prepare_data(processed_data)


def preprocess(data: DsDataList, symbol_ids: SymbolIdDict) -> Tuple[TextDataList, SymbolIdDict, SymbolsDict]:
  processed_data: List[Tuple[int, List[str], List[int], Language]] = []

  values: DsData
  for values in tqdm(data):
    symbols: List[str] = symbol_ids.get_symbols(deserialize_list(values.serialized_symbols))
    accents: List[int] = deserialize_list(values.serialized_accents)
    processed_data.append((values.entry_id, symbols, accents, values.lang))

  return _prepare_data(processed_data)


def _prepare_data(processed_data: List[Tuple[int, List[str], List[int], Language]]) -> Tuple[TextDataList, SymbolIdDict, AccentsDict, SymbolsDict]:
  result = TextDataList()
  symbol_counter = get_counter([x[1] for x in processed_data])
  symbols_dict = SymbolsDict.fromcounter(symbol_counter)
  conv = SymbolIdDict.init_from_symbols(set(symbols_dict.keys()))

  for entry_id, symbols, accent_ids, lang in processed_data:
    assert len(accent_ids) == len(symbols)
    text = SymbolIdDict.symbols_to_str(symbols)
    serialized_symbol_ids = conv.get_serialized_ids(symbols)
    serialized_accent_ids = serialize_list(accent_ids)
    data = TextData(entry_id, text, serialized_symbol_ids, serialized_accent_ids, lang)
    result.append(data)

  return result, conv, symbols_dict
