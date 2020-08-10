from dataclasses import dataclass
from typing import List
from src.core.pre.language import Language

@dataclass()
class PreData:
  name: str
  speaker_name: str
  text: str
  wav_path: str
  lang: Language

class PreDataList(List[PreData]):
  pass

if __name__ == "__main__":
  x = PreDataList([PreData("", ",", "tarei", "atrien", 1)])
  print(x)
