import os
from dataclasses import dataclass
from typing import List, Tuple

from tqdm.std import tqdm

from src.core.common.gender import Gender
from src.core.common.language import Language
from src.core.common.text import text_to_symbols
from src.core.common.utils import GenericList, cast_as, get_subfolders
from src.core.pre.parser.data import PreData, PreDataList

OATA_CSV_NAME = "data.csv"
AUDIO_FOLDER_NAME = "audio"


@dataclass
class Entry():
  entry_id: int
  text: str
  wav: str
  duration: float
  speaker: str
  gender: str
  accent: str


class Entries(GenericList[Entry]):
  def load_init(self):
    for entry in self.items():
      entry.text = str(entry.text)
      entry.speaker = str(entry.speaker)
      entry.accent = str(entry.accent)


def sort_entries_key(entry: PreData) -> Tuple[str, str]:
  return entry.speaker_name, entry.wav_path


def parse(dir_path: str) -> PreDataList:
  if not os.path.exists(dir_path):
    print("Directory not found:", dir_path)
    raise Exception()

  result = PreDataList()
  lang = Language.ENG
  tmp: List[Tuple[Tuple, PreDataList]] = []

  subfolders = get_subfolders(dir_path)
  for subfolder in tqdm(subfolders):
    data_path = os.path.join(subfolder, OATA_CSV_NAME)
    entries = cast_as(Entries.load(Entry, data_path), Entries)
    for entry in entries.items():
      gender = Gender.MALE if entry.gender == "m" else Gender.FEMALE

      symbols = text_to_symbols(entry.text, lang)
      wav_path = os.path.join(subfolder, AUDIO_FOLDER_NAME, entry.wav)
      data = PreData(
        name=entry.entry_id,
        speaker_name=entry.speaker,
        lang=lang,
        wav_path=wav_path,
        gender=gender,
        text=entry.text,
        symbols=symbols,
        accents=[entry.accent] * len(symbols),
      )
      sorting_keys = entry.speaker, subfolder, entry.entry_id
      tmp.append((sorting_keys, data))

  tmp.sort(key=lambda x: x[0])

  result = PreDataList([x for _, x in tmp])

  return result
