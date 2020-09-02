from typing import List
from src.core.common.utils import parse_json, save_json
from collections import Counter, OrderedDict

class SymbolsDict(OrderedDict): # Tuple[str, int]
  def __init__(self, *args, **kwargs):
    super(SymbolsDict, self).__init__(*args, **kwargs)

  def save(self, file_path: str):
    save_json(file_path, self)
  
  @classmethod
  def load(cls, file_path: str):
    data = parse_json(file_path)
    return cls(data)

  @classmethod
  def fromcounter(cls, counter: Counter):
    return cls(counter.most_common())
