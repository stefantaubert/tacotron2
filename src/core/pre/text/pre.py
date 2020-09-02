from collections import Counter, OrderedDict
from dataclasses import dataclass
from typing import List
from typing import OrderedDict as OrderedDictType
from typing import Tuple

from tqdm import tqdm

from src.core.common import load_csv, parse_json, save_csv, save_json, Language, SymbolIdDict, SymbolsDict, AccentsDict, deserialize_list, serialize_list, text_to_symbols, convert_to_ipa as text_convert_to_ipa, normalize as text_normalize, get_unique_items, get_counter
from src.core.pre.ds import DsData, DsDataList


@dataclass()
class TextData:
  entry_id: int
  text: str
  serialized_symbol_ids: str
  serialized_accent_ids: str
  lang: Language

class TextDataList(List[TextData]):
  def save(self, file_path: str):
    save_csv(self, file_path)

  @classmethod
  def load(cls, file_path: str):
    data = load_csv(file_path, TextData)
    return cls(data)

def convert_to_ipa(data: TextDataList, symbol_converter: SymbolIdDict, ignore_tones: bool, ignore_arcs: bool, replace_unknown_ipa_by: str) -> Tuple[TextDataList, SymbolIdDict, SymbolsDict]:
  processed_data: List[Tuple[int, List[str], List[int], Language]] = []
  
  values: TextData
  for values in tqdm(data):
    if values.lang != Language.IPA:
      orig_text = symbol_converter.get_text(values.serialized_symbol_ids)
      ipa = text_convert_to_ipa(orig_text, values.lang)
      symbols: List[str] = text_to_symbols(ipa, Language.IPA, ignore_tones, ignore_arcs, replace_unknown_ipa_by=replace_unknown_ipa_by)
      orig_accents = deserialize_list(values.serialized_accent_ids)
      if len(orig_accents):
        accents = [orig_accents[0]] * len(symbols)
      else:
        accents = []
    else:
      symbols = deserialize_list(values.serialized_symbol_ids)
      accents = deserialize_list(values.serialized_accent_ids)
    processed_data.append((values.entry_id, symbols, accents, Language.IPA))

  return _prepare_data(processed_data)

def normalize(data: TextDataList, symbol_converter: SymbolIdDict)-> Tuple[TextDataList, SymbolIdDict, SymbolsDict]:
  processed_data: List[Tuple[int, List[str], List[int], Language]] = []

  values: TextData
  for values in tqdm(data):
    orig_text = symbol_converter.get_text(values.serialized_symbol_ids)
    text = text_normalize(orig_text, values.lang)
    symbols: List[str] = text_to_symbols(text, values.lang)
    orig_accents = deserialize_list(values.serialized_accent_ids)
    if values.lang != Language.IPA:
      if len(orig_accents):
        accents = [orig_accents[0]] * len(symbols)
      else:
        accents = []
    else:
      # because no replacing was done in ipa normalization
      # maybe support remove whitespace
      accents = orig_accents

    processed_data.append((values.entry_id, symbols, accents, values.lang))

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
