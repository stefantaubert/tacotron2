from collections import OrderedDict
from typing import OrderedDict as OrderedDictType, Tuple, List, Set, Optional
from src.core.common.utils import parse_json, save_json
from src.core.common.text import get_entries_ids_dict, serialize_list


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

  def get_ids(self, accents: List[str]) -> List[int]:
    ids = [self.get_id(accent) for accent in accents]
    return ids

  def get_serialized_ids(self, accents: List[str]) -> str:
    ids = self.get_ids(accents)
    return serialize_list(ids)

  @classmethod
  def load(cls, file_path: str):
    data = parse_json(file_path)
    return cls(data)
 
  @classmethod
  def init_from_accents(cls, accents: Set[str]):
    ids_to_accents = get_entries_ids_dict(accents)
    return cls(ids_to_accents)
