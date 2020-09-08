from typing import List
from typing import OrderedDict as OrderedDictType
from typing import Set, Union

from src.core.common.globals import PADDING_ACCENT
from src.core.common.text import (deserialize_list, get_entries_ids_dict_order,
                                  serialize_list)
from src.core.common.utils import parse_json, save_json


class AccentsDict():
  def __init__(self, ids_to_accents: OrderedDictType[int, str]):
    super().__init__()
    self._ids_to_accents = ids_to_accents

  def save(self, file_path: str):
    save_json(file_path, self._ids_to_accents)

  def get_id(self, accent: str) -> int:
    assert accent in self._ids_to_accents.keys()
    return self._ids_to_accents[accent]

  def get_accent(self, accent_id: str) -> str:
    assert accent_id in self._ids_to_accents.values()
    for accent, a_id in self._ids_to_accents.items():
      if a_id == accent_id:
        return accent
    assert False

  def get_all_accents(self) -> Set[str]:
    return set(self._ids_to_accents.keys())

  def get_ids(self, accents: List[str]) -> List[int]:
    ids = [self.get_id(accent) for accent in accents]
    return ids

  def get_accents(self, accent_ids: Union[str, List[int]]) -> List[str]:
    if isinstance(accent_ids, str):
      accent_ids = deserialize_list(accent_ids)
    elif not isinstance(accent_ids, list):
      assert False
    accents = [self.get_accent(accent_id) for accent_id in accent_ids]
    return accents

  def get_serialized_ids(self, accents: List[str]) -> str:
    ids = self.get_ids(accents)
    return serialize_list(ids)

  def __len__(self):
    return len(self._ids_to_accents)

  @classmethod
  def load(cls, file_path: str):
    data = parse_json(file_path)
    return cls(data)

  @classmethod
  def init_from_accents(cls, accents: Set[str]):
    unique_entries = list(sorted(accents))
    ids_to_accents = get_entries_ids_dict_order(unique_entries)
    return cls(ids_to_accents)

  @classmethod
  def init_from_accents_with_pad(cls, accents: Set[str], pad_accent: str = PADDING_ACCENT):
    unique_entries = list(sorted(accents - {pad_accent}))
    final_accents = [pad_accent] + unique_entries
    ids_to_accents = get_entries_ids_dict_order(final_accents)
    return cls(ids_to_accents)
