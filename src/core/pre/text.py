from collections import Counter, OrderedDict
from dataclasses import dataclass
from typing import List
from typing import OrderedDict as OrderedDictType
from typing import Tuple

import epitran
from tqdm import tqdm

from src.core.common.utils import load_csv, parse_json, save_csv, save_json
from src.core.pre.ds import DsData, DsDataList
from src.core.pre.language import Language
from src.text.adjustments import normalize_text
from src.text.chn_tools import chn_to_ipa
from src.text.ipa2symb import extract_from_sentence
from src.text.symbol_converter import SymbolConverter


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

def convert_to_ipa(data: TextDataList, ignore_tones: bool, ignore_arcs: bool) -> Tuple[TextDataList, SymbolConverter, SymbolsDict]:
  result = TextDataList()
  entry_symbols = []
  all_symbols = []
  epi = epitran.Epitran('eng-Latn')
  
  values: TextData
  for values in tqdm(data):
    if values.lang == Language.CHN:
      ipa = chn_to_ipa(values.text, add_period=True)
    elif values.lang == Language.ENG:
      ipa = epi.transliterate(values.text)
    else: assert False
    symbols: List[str] = extract_from_sentence(ipa, ignore_tones, ignore_arcs)
    entry_symbols.append(symbols)
    all_symbols.extend(symbols)
    ipa_symbols = ''.join(symbols)
    result.append(TextData(values.entry_id, ipa_symbols, "", Language.IPA))

  symbol_counter = Counter(all_symbols)
  symbols_dict = SymbolsDict.fromcounter(symbol_counter)
  conv = SymbolConverter.init_from_symbols(set(symbols_dict.keys()))

  for i, symbols in enumerate(entry_symbols):
    symbol_ids = conv.symbols_to_ids(symbols, add_eos=True, replace_unknown_with_pad=True)
    result[i].serialized_symbol_ids = SymbolConverter.serialize_symbol_ids(symbol_ids)

  return result, conv, symbols_dict

def normalize(data: TextDataList)-> Tuple[TextDataList, SymbolConverter, SymbolsDict]:
  result = TextDataList()
  entry_symbols = []
  all_symbols = []
  
  values: TextData
  for values in tqdm(data):
    if values.lang == Language.ENG:
      text = normalize_text(values.text)
    elif values.lang == Language.CHN:
      text = values.text
    else: assert False
    symbols: List[str] = list(text)
    entry_symbols.append(symbols)
    all_symbols.extend(symbols)
    result.append(TextData(values.entry_id, values.text, "", values.lang))

  symbol_counter = Counter(all_symbols)
  symbols_dict = SymbolsDict.fromcounter(symbol_counter)
  conv = SymbolConverter.init_from_symbols(set(symbols_dict.keys()))

  for i, symbols in enumerate(entry_symbols):
    symbol_ids = conv.symbols_to_ids(symbols, add_eos=True, replace_unknown_with_pad=True)
    result[i].serialized_symbol_ids = serialize_symbol_ids(symbol_ids)

  return result, conv, symbols_dict

def preprocess(data: DsDataList) -> Tuple[TextDataList, SymbolConverter, SymbolsDict]:
  result = TextDataList()
  entry_symbols = []
  all_symbols = []
  
  values: DsData
  for values in tqdm(data):
    symbols: List[str] = list(values.text)
    entry_symbols.append(symbols)
    all_symbols.extend(symbols)
    result.append(TextData(values.entry_id, values.text, "", values.lang))

  symbol_counter = Counter(all_symbols)
  symbols_dict = SymbolsDict.fromcounter(symbol_counter)
  conv = SymbolConverter.init_from_symbols(set(symbols_dict.keys()))

  for i, symbols in enumerate(entry_symbols):
    symbol_ids = conv.symbols_to_ids(symbols, add_eos=True, replace_unknown_with_pad=True)
    result[i].serialized_symbol_ids = serialize_symbol_ids(symbol_ids)

  return result, conv, symbols_dict
