import os
from collections import Counter
from dataclasses import dataclass
from typing import List, OrderedDict, Tuple

from src.core.common import parse_json, save_json, Language, SpeakersDict, SpeakersLogDict, AccentsDict, SymbolIdDict, remove_duplicates_list_orderpreserving, GenericList
from src.core.pre.parser import (PreData, PreDataList, dl_kaldi, dl_ljs,
                                 dl_thchs, parse_ljs, parse_thchs,
                                 parse_thchs_kaldi)


@dataclass()
class DsData:
  entry_id: int
  basename: str
  speaker_name: str
  speaker_id: int
  text: str
  serialized_symbols: str
  serialized_accents: str
  wav_path: str
  lang: Language

  def get_speaker_name(self):
    return str(self.speaker_name)


class DsDataList(GenericList[DsData]):
  pass


def _preprocess_core(dir_path: str, auto_dl: bool, dl_func, parse_func) -> Tuple[
        SpeakersDict, SpeakersLogDict, DsDataList, SymbolIdDict, AccentsDict]:
  if not os.path.isdir(dir_path) and auto_dl:
    dl_func(dir_path)
  data = parse_func(dir_path)
  speakers, speakers_log = _get_all_speakers(data)
  accents = _get_all_accents(data)
  symbols = _get_symbols_id_dict(data)
  ds_data = _get_ds_data(data, speakers, accents, symbols)
  return speakers, speakers_log, symbols, accents, ds_data


def thchs_preprocess(dir_path: str, auto_dl: bool) -> Tuple[
        SpeakersDict, SpeakersLogDict, DsDataList, SymbolIdDict, AccentsDict]:
  return _preprocess_core(dir_path, auto_dl, dl_thchs, parse_thchs)


def ljs_preprocess(dir_path: str, auto_dl: bool) -> Tuple[
  SpeakersDict, SpeakersLogDict, DsDataList, SymbolIdDict, AccentsDict]:
  return _preprocess_core(dir_path, auto_dl, dl_ljs, parse_ljs)


def thchs_kaldi_preprocess(dir_path: str, auto_dl: bool) -> Tuple[SpeakersDict, SpeakersLogDict, DsDataList, SymbolIdDict, AccentsDict]:
  return _preprocess_core(dir_path, auto_dl, dl_kaldi, parse_thchs_kaldi)


def _get_all_speakers(l: PreDataList) -> Tuple[SpeakersDict, SpeakersLogDict]:
  x: PreData
  all_speakers: List[str] = [x.speaker_name for x in l]
  all_speakers_count = Counter(all_speakers)
  speakers_log = SpeakersLogDict.fromcounter(all_speakers_count)
  all_speakers = remove_duplicates_list_orderpreserving(all_speakers)
  speakers_dict = SpeakersDict.fromlist(all_speakers)
  return speakers_dict, speakers_log


def _get_all_accents(l: PreDataList) -> AccentsDict:
  accents = set()
  x: PreData
  for x in l:
    accents = accents.union(set(x.accents))
  return AccentsDict.init_from_accents(accents)


def _get_symbols_id_dict(l: PreDataList) -> SymbolIdDict:
  symbols = set()
  x: PreData
  for x in l:
    symbols = symbols.union(set(x.symbols))
  return SymbolIdDict.init_from_symbols(symbols)


def _get_ds_data(l: PreDataList, speakers_dict: SpeakersDict, accents: AccentsDict, symbols: SymbolIdDict) -> DsDataList:
  values: PreData
  result = [DsData(
    i,
    values.name,
    values.speaker_name,
    speakers_dict[values.speaker_name],
    values.text,
    symbols.get_serialized_ids(values.symbols),
    accents.get_serialized_ids(values.accents),
    values.wav_path,
    values.lang
  ) for i, values in enumerate(l)]
  return DsDataList(result)
