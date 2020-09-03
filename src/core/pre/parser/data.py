from dataclasses import dataclass
from src.core.common import GenericList
from typing import List
from src.core.common import Language


@dataclass()
class PreData:
  name: str
  speaker_name: str
  text: str
  wav_path: str
  symbols: List[str]
  accents: List[str]
  lang: Language

class PreDataList(GenericList[PreData]):
  pass

if __name__ == "__main__":
  x = PreDataList([PreData("", ",", "tarei", "atrien", Language.CHN)])
  print(x)
  print(type(x))
