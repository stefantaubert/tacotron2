import os
from collections import Counter
from dataclasses import dataclass
from typing import Any, List, Optional, Set, Tuple

from src.core.common.accents_dict import AccentsDict
from src.core.common.gender import Gender
from src.core.common.language import Language
from src.core.common.speakers_dict import SpeakersDict, SpeakersLogDict
from src.core.common.symbol_id_dict import SymbolIdDict
from src.core.common.utils import (GenericList,
                                   remove_duplicates_list_orderpreserving)
from src.core.pre.parser.arctic import download as dl_arctic
from src.core.pre.parser.arctic import parse as parse_arctic
from src.core.pre.parser.custom import parse as parse_custom
from src.core.pre.parser.data import PreDataList
from src.core.pre.parser.libritts import download as dl_libritts
from src.core.pre.parser.libritts import parse as parse_libritts
from src.core.pre.parser.ljs import download as dl_ljs
from src.core.pre.parser.ljs import parse as parse_ljs
from src.core.pre.parser.thchs import download as dl_thchs
from src.core.pre.parser.thchs import parse as parse_thchs
from src.core.pre.parser.thchs_kaldi import download as dl_kaldi
from src.core.pre.parser.thchs_kaldi import parse as parse_thchs_kaldi


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
  gender: Gender

  def load_init(self):
    self.lang = Language(self.lang)
    self.gender = Gender(self.gender)
    self.speaker_name = str(self.speaker_name)


class DsDataList(GenericList[DsData]):
  def load_init(self):
    for item in self.items():
      item.load_init()
from functools import partial

def _preprocess_core(dir_path: str, auto_dl: bool, dl_func: Optional[Any], parse_func) -> Tuple[
        SpeakersDict, SpeakersLogDict, DsDataList, SymbolIdDict, AccentsDict]:
  if not os.path.isdir(dir_path) and auto_dl and dl_func is not None:
    dl_func(dir_path)

  data = parse_func(dir_path)
  speakers, speakers_log = _get_all_speakers(data)
  accents = _get_all_accents(data)
  symbols = _get_symbols_id_dict(data)
  ds_data = _get_ds_data(data, speakers, accents, symbols)
  return speakers, speakers_log, symbols, accents, ds_data


def get_speaker_examples(data: DsDataList) -> DsDataList:
  processed_speakers: Set[str] = set()
  result = DsDataList()
  for values in data.items(True):
    if values.speaker_name not in processed_speakers:
      processed_speakers.add(values.speaker_name)
      result.append(values)
  return result


def thchs_preprocess(dir_path: str, auto_dl: bool) -> Tuple[
        SpeakersDict, SpeakersLogDict, DsDataList, SymbolIdDict, AccentsDict]:
  return _preprocess_core(dir_path, auto_dl, dl_thchs, parse_thchs)


def custom_preprocess(dir_path: str, auto_dl: bool) -> Tuple[
  SpeakersDict, SpeakersLogDict, DsDataList, SymbolIdDict, AccentsDict]:
  return _preprocess_core(dir_path, auto_dl, None, parse_custom)


def libritts_preprocess(dir_path: str, auto_dl: bool) -> Tuple[
  SpeakersDict, SpeakersLogDict, DsDataList, SymbolIdDict, AccentsDict]:
  return _preprocess_core(dir_path, auto_dl, dl_libritts, parse_libritts)


def arctic_preprocess(dir_path: str, auto_dl: bool) -> Tuple[
  SpeakersDict, SpeakersLogDict, DsDataList, SymbolIdDict, AccentsDict]:
  return _preprocess_core(dir_path, auto_dl, dl_arctic, parse_arctic)


def ljs_preprocess(dir_path: str, auto_dl: bool) -> Tuple[
  SpeakersDict, SpeakersLogDict, DsDataList, SymbolIdDict, AccentsDict]:
  return _preprocess_core(dir_path, auto_dl, dl_ljs, parse_ljs)


def thchs_kaldi_preprocess(dir_path: str, auto_dl: bool) -> Tuple[SpeakersDict, SpeakersLogDict, DsDataList, SymbolIdDict, AccentsDict]:
  return _preprocess_core(dir_path, auto_dl, dl_kaldi, parse_thchs_kaldi)


def _get_all_speakers(l: PreDataList) -> Tuple[SpeakersDict, SpeakersLogDict]:
  all_speakers: List[str] = [x.speaker_name for x in l.items()]
  all_speakers_count = Counter(all_speakers)
  speakers_log = SpeakersLogDict.fromcounter(all_speakers_count)
  all_speakers = remove_duplicates_list_orderpreserving(all_speakers)
  speakers_dict = SpeakersDict.fromlist(all_speakers)
  return speakers_dict, speakers_log


def _get_all_accents(l: PreDataList) -> AccentsDict:
  accents = set()
  for x in l.items():
    accents = accents.union(set(x.accents))
  return AccentsDict.init_from_accents(accents)


def _get_symbols_id_dict(l: PreDataList) -> SymbolIdDict:
  symbols = set()
  for x in l.items():
    symbols = symbols.union(set(x.symbols))
  return SymbolIdDict.init_from_symbols(symbols)


def _get_ds_data(l: PreDataList, speakers_dict: SpeakersDict, accents: AccentsDict, symbols: SymbolIdDict) -> DsDataList:
  result = [DsData(
    entry_id=i,
    basename=values.name,
    speaker_name=values.speaker_name,
    speaker_id=speakers_dict[values.speaker_name],
    text=values.text,
    serialized_symbols=symbols.get_serialized_ids(values.symbols),
    serialized_accents=accents.get_serialized_ids(values.accents),
    wav_path=values.wav_path,
    lang=values.lang,
    gender=values.gender
  ) for i, values in enumerate(l.items())]
  return DsDataList(result)
