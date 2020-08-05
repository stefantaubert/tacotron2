from dataclasses import dataclass
from typing import List

@dataclass()
class PreData:
  name: str
  speaker_name: str
  text: str
  wav_path: str

PreDataList = List[PreData]
