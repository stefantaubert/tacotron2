from dataclasses import dataclass
from typing import List
from src.common.utils import load_csv, save_csv

@dataclass()
class WavData:
  i: int
  wav: str
  duration: float
  sr: int

  def __repr__(self):
    return str(self.i)
  
class WavDataList(List[WavData]):
  def save(self, file_path: str):
    save_csv(self, file_path)

  @classmethod
  def load(cls, file_path: str):
    data = load_csv(file_path, WavData)
    return cls(data)
