from dataclasses import dataclass
from logging import Logger
from typing import Dict, List, Optional, Tuple

import pandas as pd
from numpy.core.fromnumeric import mean
from src.core.common.globals import PADDING_SYMBOL
from src.core.common.utils import GenericList, get_counter
from src.core.pre.ds import DsDataList
from text_utils import (AccentsDict, EngToIpaMode, Language, SymbolIdDict,
                        SymbolsDict, deserialize_list, serialize_list,
                        symbols_normalize, symbols_to_ipa)


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


def log_stats(ds_data: DsDataList, text_data: TextDataList, logger: Logger):
  stats: List[str, int, float, float, float] = []
  text_lengths = [len(deserialize_list(x.serialized_symbol_ids)) for x in text_data.items()]
  stats.append((
    "Overall",
    len(text_lengths),
    min(text_lengths),
    max(text_lengths),
    mean(text_lengths),
    sum(text_lengths),
  ))

  speakers_text_lengths: List[int, List[float]] = {}
  speaker_names: Dict[int, str] = {}
  for ds_entry, text_entry in zip(ds_data.items(), text_data.items()):
    if ds_entry.speaker_id not in speakers_text_lengths:
      speakers_text_lengths[ds_entry.speaker_id] = []
      speaker_names[ds_entry.speaker_id] = ds_entry.speaker_name
    speakers_text_lengths[ds_entry.speaker_id].append(
      len(deserialize_list(text_entry.serialized_symbol_ids)))

  for k, speaker_text_lengths in speakers_text_lengths.items():
    stats.append((
      f"{speaker_names[k]} ({k})",
      len(speaker_text_lengths),
      min(speaker_text_lengths),
      max(speaker_text_lengths),
      mean(speaker_text_lengths),
      sum(speaker_text_lengths),
    ))

  stats.sort(key=lambda x: (x[-1]), reverse=True)
  stats_csv = pd.DataFrame(stats, columns=[
    "Speaker",
    "# Entries",
    "# Min",
    "# Max",
    "# Avg",
    "# Total",
  ])

  with pd.option_context(
    'display.max_rows', None,
    'display.max_columns', None,
    'display.width', None,
    'display.precision', 0,
  ):  # more options can be specified also
    print(stats_csv)


def convert_to_ipa(data: TextDataList, symbol_converter: SymbolIdDict, ignore_tones: bool, ignore_arcs: bool, mode: Optional[EngToIpaMode], logger: Logger) -> Tuple[TextDataList, SymbolIdDict, SymbolsDict]:
  processed_data: List[Tuple[int, List[str], List[int], Language]] = []

  for values in data.items(True):
    if values.lang == Language.ENG and mode is None:
      ex = "Please specify the ipa conversion mode."
      logger.exception(ex)
      raise Exception(ex)
    new_symbols, new_accent_ids = symbols_to_ipa(
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
