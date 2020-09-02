from src.core.common.utils import parse_json, save_json
from collections import Counter, OrderedDict


class SpeakersDict(OrderedDict):  # [str, int]
  def save(self, file_path: str):
    save_json(file_path, self)

  def get_speakers(self):
    return list(self.keys())

  @classmethod
  def load(cls, file_path: str):
    data = parse_json(file_path)
    return cls(data)

  @classmethod
  def fromlist(cls, lst: list):
    res = [(x, i) for i, x in enumerate(lst)]
    return cls(cls, res)


class SpeakersLogDict(OrderedDict):  # [str, int]
  def save(self, file_path: str):
    save_json(file_path, self)

  @classmethod
  def fromcounter(cls, counter: Counter):
    return cls(counter.most_common())
