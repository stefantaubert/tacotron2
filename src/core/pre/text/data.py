from dataclasses import dataclass
from typing import List
from src.core.pre.language import Language
from src.common.utils import load_csv, save_csv

@dataclass()
class TextData:
  i: int
  text: str
  serialized_symbol_ids: str
  lang: Language

class TextDataList(List[TextData]):
  def save(self, file_path: str):
    save_csv(self, file_path)

  @classmethod
  def load(cls, file_path: str):
    data = load_csv(file_path, TextData)
    return cls(data)
