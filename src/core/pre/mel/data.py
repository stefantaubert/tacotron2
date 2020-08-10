from dataclasses import dataclass
from typing import List
from src.common.utils import load_csv, save_csv

@dataclass()
class MelData:
  i: int
  mel_path: str
  duration: float
  sr: int

class MelDataList(List[MelData]):
  def save(self, file_path: str):
    save_csv(self, file_path)

  @classmethod
  def load(cls, file_path: str):
    data = load_csv(file_path, MelData)
    return cls(data)
