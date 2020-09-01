from collections import Counter, OrderedDict
from dataclasses import dataclass
from typing import List
from typing import OrderedDict as OrderedDictType
from typing import Tuple

import epitran
from tqdm import tqdm

from src.core.common import load_csv, parse_json, save_csv, save_json
from src.core.pre.ds import DsData, DsDataList
from src.core.common import Language
from src.core.pre.text.adjustments import normalize_text
from src.core.pre.text.chn_tools import chn_to_ipa
from src.core.pre.text.ipa2symb import extract_from_sentence
from src.core.pre.text.symbol_id_dict import SymbolIdDict


class SymbolsDict(OrderedDictType[str, int]):
  def save(self, file_path: str):
    save_json(file_path, self)
  
  @classmethod
  def load(cls, file_path: str):
    data = parse_json(file_path)
    return cls(data)

  @classmethod
  def fromcounter(cls, counter: Counter):
    return cls(counter.most_common())

@dataclass()
class TextData:
  entry_id: int
  text: str
  serialized_symbol_ids: str
  lang: Language

class TextDataList(List[TextData]):
  def save(self, file_path: str):
    save_csv(self, file_path)

  @classmethod
  def load(cls, file_path: str):
    data = load_csv(file_path, TextData)
    return cls(data)

def convert_to_ipa(data: TextDataList, symbol_converter: SymbolIdDict, ignore_tones: bool, ignore_arcs: bool) -> Tuple[TextDataList, SymbolIdDict, SymbolsDict]:
  processed_data: List[Tuple[int, List[str], Language]] = []
  epi = epitran.Epitran('eng-Latn')
  
  values: TextData
  for values in tqdm(data):
    orig_text = symbol_converter.serialized_symbol_ids_to_text(values.serialized_symbol_ids)

    if values.lang == Language.CHN:
      ipa = chn_to_ipa(orig_text, add_period=True)
    elif values.lang == Language.ENG:
      ipa = epi.transliterate(orig_text)
    else:
      assert False

    symbols: List[str] = extract_from_sentence(ipa, ignore_tones, ignore_arcs, replace_unknown_ipa_by="_")
    processed_data.append((values.entry_id, symbols, Language.IPA))

  return _prepare_data(processed_data)

def normalize(data: TextDataList, symbol_converter: SymbolIdDict)-> Tuple[TextDataList, SymbolIdDict, SymbolsDict]:
  processed_data: List[Tuple[int, List[str], Language]] = []

  values: TextData
  for values in tqdm(data):
    orig_text = symbol_converter.serialized_symbol_ids_to_text(values.serialized_symbol_ids)

    if values.lang == Language.ENG:
      text = normalize_text(orig_text)
    elif values.lang == Language.CHN:
      text = orig_text
    else:
      assert False

    symbols: List[str] = list(text)
    processed_data.append((values.entry_id, symbols, values.lang))

  return _prepare_data(processed_data)

def preprocess(data: DsDataList) -> Tuple[TextDataList, SymbolIdDict, SymbolsDict]:
  processed_data: List[Tuple[int, List[str], Language]] = []
  
  values: DsData
  for values in tqdm(data):
    symbols: List[str] = list(values.text)
    processed_data.append((values.entry_id, symbols, values.lang))

  return _prepare_data(processed_data)

def _prepare_data(processed_data: List[Tuple[int, List[str], Language]]):
  all_symbols = []
  result = TextDataList()

  for _, symbols, _ in processed_data:
    all_symbols.extend(symbols)

  symbol_counter = Counter(all_symbols)
  symbols_dict = SymbolsDict.fromcounter(symbol_counter)
  conv = SymbolIdDict.init_from_symbols(set(symbols_dict.keys()))

  for entry_id, symbols, lang in processed_data:
    symbol_ids = conv.get_ids(symbols)
    text = SymbolIdDict.symbols_to_str(symbols)
    serialized_symbol_ids = SymbolIdDict.serialize_symbol_ids(symbol_ids)
    data = TextData(entry_id, text, serialized_symbol_ids, lang)
    result.append(data)

  return result, conv, symbols_dict