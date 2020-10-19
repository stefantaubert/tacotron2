from collections import Counter, OrderedDict
from typing import List
from typing import OrderedDict as OrderedDictType
from typing import Set

from src.core.common.text import switch_keys_with_values
from src.core.common.utils import parse_json, save_json


class SpeakersDict(OrderedDict):  # [str, int]
  def save(self, file_path: str):
    save_json(file_path, self.raw())

  def get_all_speakers(self) -> List[str]:
    return list(self.keys())

  def get_all_speaker_ids(self) -> List[int]:
    return list(self.values())

  def remove_ids(self, ids: Set[int]) -> None:
    pop_keys: Set[str] = set()
    for speaker_name, speaker_id in self.items():
      if speaker_id in ids:
        pop_keys |= {speaker_name}

    for pop_key in pop_keys:
      self.pop(pop_key)

  def get_id(self, speaker: str) -> int:
    result = self[speaker]
    return result

  def get_speaker(self, speaker_id: int) -> str:
    result = switch_keys_with_values(self)[speaker_id]
    return result

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
    return cls(res)


class SpeakersLogDict(OrderedDict):  # [str, int]
  def save(self, file_path: str):
    save_json(file_path, self)

  @classmethod
  def fromcounter(cls, counter: Counter):
    return cls(counter.most_common())
