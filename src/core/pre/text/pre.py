from dataclasses import dataclass
from logging import Logger
from src.core.common.globals import PADDING_SYMBOL
from typing import List, Optional, Tuple

from text_utils.text import EngToIpaMode

from src.core.common.accents_dict import AccentsDict
from src.core.common.language import Language
from src.core.common.symbol_id_dict import SymbolIdDict
from src.core.common.symbols_dict import SymbolsDict
from src.core.common.text import (deserialize_list,
                                  serialize_list)
from src.core.common.utils import GenericList, get_counter
from src.core.pre.ds import DsDataList
from src.core.pre.text.utils import symbols_convert_to_ipa, symbols_normalize


@dataclass()
class TextData:
  entry_id: int
  text: str
  serialized_symbol_ids: str
  serialized_accent_ids: str
  lang: Language

  def load_init(self):
    self.lang = Language(self.lang)


class TextDataList(GenericList[TextData]):
  def load_init(self):
    for item in self.items():
      item.load_init()


def convert_to_ipa(data: TextDataList, symbol_converter: SymbolIdDict, ignore_tones: bool, ignore_arcs: bool, mode: Optional[EngToIpaMode], logger: Logger) -> Tuple[TextDataList, SymbolIdDict, SymbolsDict]:
  processed_data: List[Tuple[int, List[str], List[int], Language]] = []

  for values in data.items(True):
    if values.lang == Language.ENG and mode is None:
      ex = "Please specify the ipa conversion mode."
      logger.exception(ex)
      raise Exception(ex)
    new_symbols, new_accent_ids = symbols_convert_to_ipa(
      symbols=symbol_converter.get_symbols(values.serialized_symbol_ids),
      lang=values.lang,
      accent_ids=deserialize_list(values.serialized_accent_ids),
      ignore_arcs=ignore_arcs,
      ignore_tones=ignore_tones,
      mode=mode,
      replace_unknown_with=PADDING_SYMBOL,
      logger=logger,
    )
    processed_data.append((values.entry_id, new_symbols, new_accent_ids, Language.IPA))

  return _prepare_data(processed_data)


def normalize(data: TextDataList, symbol_converter: SymbolIdDict, logger: Logger) -> Tuple[TextDataList, SymbolIdDict, SymbolsDict]:
  processed_data: List[Tuple[int, List[str], List[int], Language]] = []

  for values in data.items(True):
    new_symbols, new_accent_ids = symbols_normalize(
      symbols=symbol_converter.get_symbols(values.serialized_symbol_ids),
      lang=values.lang,
      accent_ids=deserialize_list(values.serialized_accent_ids),
      logger=logger,
    )

    processed_data.append((values.entry_id, new_symbols, new_accent_ids, values.lang))

  return _prepare_data(processed_data)


def preprocess(data: DsDataList, symbol_ids: SymbolIdDict) -> Tuple[TextDataList, SymbolIdDict, SymbolsDict]:
  processed_data: List[Tuple[int, List[str], List[int], Language]] = []

  for values in data.items(True):
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
    text = SymbolIdDict.symbols_to_text(symbols)
    serialized_symbol_ids = conv.get_serialized_ids(symbols)
    serialized_accent_ids = serialize_list(accent_ids)
    data = TextData(entry_id, text, serialized_symbol_ids, serialized_accent_ids, lang)
    result.append(data)

  return result, conv, symbols_dict
