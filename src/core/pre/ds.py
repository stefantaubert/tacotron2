from src.core.pre.parser import PreDataList, PreData
from typing import List, OrderedDict, Tuple
from collections import Counter
from dataclasses import dataclass
from src.core.pre.language import Language
from src.common.utils import load_csv, save_csv, save_json, parse_json
from collections import Counter

class SpeakersDict(OrderedDict[str, int]):
  def save(self, file_path: str):
    save_json(file_path, self)
  
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

class DsDataList(List[DsData]):
  def save(self, file_path: str):
    save_csv(self, file_path)

  @classmethod
  def load(cls, file_path: str):
    data = load_csv(file_path, DsData)
    return cls(data)

def get_all_speakers(l: PreDataList) -> Tuple[SpeakersDict, SpeakersLogDict]:
  all_speakers: List[str] = [x.speaker_name for x in l]
  all_speakers_count = Counter(all_speakers)
  speakers_log = SpeakersLogDict.fromcounter(all_speakers_count)
  all_speakers = __remove_duplicates(all_speakers)
  x: PreData
  speakers_dict = SpeakersDict.fromlist(all_speakers)
  return speakers_dict, speakers_log

def get_ds_data(l: PreDataList, speakers_dict: SpeakersDict) -> DsDataList:
  values: PreData
  result = [DsData(i, values.name, values.speaker_name, speakers_dict[values.speaker_name], values.text, values.wav_path, values.lang) for i, values in enumerate(l)]
  return DsDataList(result)

def __remove_duplicates(l: List[str]) -> List[str]:
  result = []
  for x in l:
    if x not in result:
      result.append(x)
  return result
