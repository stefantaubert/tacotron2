from collections import Counter, OrderedDict
from typing import OrderedDict as OrderedDictType

from src.core.common.utils import parse_json, save_json


class SpeakersDict(OrderedDict):  # [str, int]
  def save(self, file_path: str):
    save_json(file_path, self.raw())

  def get_speakers(self):
    return list(self.keys())

  def id_exists(self, speaker_id: int) -> bool:
    return speaker_id in self.values()

  def raw(self) -> OrderedDictType[str, int]:
    return OrderedDict(self)

  @classmethod
  def from_raw(cls, raw: OrderedDictType[str, int]):
    return cls(raw)

  @classmethod
  def load(cls, file_path: str):
    data = parse_json(file_path)
    loaded = OrderedDict(data.items())
    return cls.from_raw(loaded)

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
