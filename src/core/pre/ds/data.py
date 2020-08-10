from dataclasses import dataclass
from typing import List, OrderedDict
from src.core.pre.language import Language
from src.common.utils import load_csv, save_csv, save_json, parse_json
from src.core.pre.parser.data import PreDataList, PreData

class SpeakersDict(OrderedDict[str, int]):
  def save(self, file_path: str):
    save_json(file_path, self)
  
  @classmethod
  def load(cls, file_path: str):
    data = parse_json(file_path)
    return cls(data)

  @classmethod
  def fromlist(cls, l: PreDataList):
    all_speakers: List[str] = [x.speaker_name for x in l]
    all_speakers = remove_duplicates(all_speakers)
    x: PreData
    res = [(x, i) for i, x in enumerate(all_speakers)]
    return cls(res)

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

  @classmethod
  def fromlist(cls, data: PreDataList, speakers_dict: SpeakersDict):
    values: PreData
    result = [DsData(i, values.name, values.speaker_name, speakers_dict[values.speaker_name], values.text, values.wav_path, values.lang) for i, values in enumerate(data)]
    return cls(result)

def remove_duplicates(l: List[str]) -> List[str]:
  result = []
  for x in l:
    if x not in result:
      result.append(x)
  return result
