from typing import OrderedDict
from src.core.common.utils import parse_json, save_json
from collections import Counter

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
