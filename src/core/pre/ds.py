import os
from collections import Counter
from dataclasses import dataclass
from typing import List, OrderedDict, Tuple

from src.core.common import load_csv, parse_json, save_csv, save_json
from src.core.common import Language
from src.core.pre.parser import (PreData, PreDataList, dl_kaldi, dl_ljs,
                                 dl_thchs, parse_ljs, parse_thchs,
                                 parse_thchs_kaldi)


class SpeakersDict(OrderedDict[str, int]):
  def save(self, file_path: str):
    save_json(file_path, self)
  
  def get_speakers(self):
    return list(self.keys())

  @classmethod
  def load(cls, file_path: str):
    data = parse_json(file_path)
    return cls(data)

  @classmethod
  def fromlist(cls, l: list):
    res = [(x, i) for i, x in enumerate(l)]
    return cls(res)

class SpeakersLogDict(OrderedDict[str, int]):
  def save(self, file_path: str):
    save_json(file_path, self)

  @classmethod
  def fromcounter(cls, c: Counter):
    return cls(c.most_common())

@dataclass()
class DsData:
  entry_id: int
  basename: str
  speaker_name: str
  speaker_id: int
  text: str
  wav_path: str
  lang: Language

  def get_speaker_name(self):
    return str(self.speaker_name)

class DsDataList(List[DsData]):
  def save(self, file_path: str):
    save_csv(self, file_path)

  @classmethod
  def load(cls, file_path: str):
    data = load_csv(file_path, DsData)
    return cls(data)

def _preprocess_core(dir_path: str, auto_dl: bool, dl_func, parse_func) -> Tuple[SpeakersDict, SpeakersLogDict, DsDataList]:
  if not os.path.isdir(dir_path) and auto_dl:
    dl_func(dir_path)
  data = parse_func(dir_path)
  speakers, speakers_log = _get_all_speakers(data)
  ds_data = _get_ds_data(data, speakers)
  return speakers, speakers_log, ds_data

def thchs_preprocess(dir_path: str, auto_dl: bool) -> Tuple[SpeakersDict, SpeakersLogDict, DsDataList]:
  return _preprocess_core(dir_path, auto_dl, dl_thchs, parse_thchs)

def ljs_preprocess(dir_path: str, auto_dl: bool) -> Tuple[SpeakersDict, SpeakersLogDict, DsDataList]:
  return _preprocess_core(dir_path, auto_dl, dl_ljs, parse_ljs)

def thchs_kaldi_preprocess(dir_path: str, auto_dl: bool) -> Tuple[SpeakersDict, SpeakersLogDict, DsDataList]:
  return _preprocess_core(dir_path, auto_dl, dl_kaldi, parse_thchs_kaldi)

def _get_all_speakers(l: PreDataList) -> Tuple[SpeakersDict, SpeakersLogDict]:
  x: PreData
  all_speakers: List[str] = [x.speaker_name for x in l]
  all_speakers_count = Counter(all_speakers)
  speakers_log = SpeakersLogDict.fromcounter(all_speakers_count)
  all_speakers = _remove_duplicates(all_speakers)
  speakers_dict = SpeakersDict.fromlist(all_speakers)
  return speakers_dict, speakers_log

def _get_ds_data(l: PreDataList, speakers_dict: SpeakersDict) -> DsDataList:
  values: PreData
  result = [DsData(i, values.name, values.speaker_name, speakers_dict[values.speaker_name], values.text, values.wav_path, values.lang) for i, values in enumerate(l)]
  return DsDataList(result)

def _remove_duplicates(l: List[str]) -> List[str]:
  result = []
  for x in l:
    if x not in result:
      result.append(x)
  return result
