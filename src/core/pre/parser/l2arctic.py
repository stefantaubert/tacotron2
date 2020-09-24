import os
from logging import Logger, getLogger
from typing import List, Tuple

from tqdm import tqdm

from src.core.common.gender import Gender
from src.core.common.language import Language
from src.core.common.text import text_to_symbols
from src.core.common.utils import (download_tar, get_basename, get_filenames,
                                   get_filepaths, get_subfolders, read_lines,
                                   read_text)
from src.core.pre.parser.data import PreData, PreDataList


def download(dir_path: str):
  pass


def parse(dir_path: str, logger: Logger = getLogger()) -> PreDataList:
  if not os.path.exists(dir_path):
    logger.exception(f"Directory not found: {dir_path}!")
    raise Exception()

  readme_path = os.path.join(dir_path, "README.md")
  readme = read_lines(readme_path)
  readme = readme[34:58]
  speakers_dict = {}
  for speaker_details in readme:
    name, gender, accent, _, _ = speaker_details[1:-1].split("|")
    speakers_dict[name] = gender, accent

  speaker_folders = get_subfolders(dir_path)
  lang = Language.ENG

  entries = PreDataList()

  logger.info("Parsing files...")
  for speaker_folder in tqdm(speaker_folders):
    speaker_name = get_basename(speaker_folder)
    if speaker_name not in speakers_dict.keys():
      logger.info(f"Skipping {speaker_name}")
      continue
    wavs = get_filepaths(os.path.join(speaker_folder, "wav"))
    annotations = get_filepaths(os.path.join(speaker_folder, "annotation"))  # only 150
    textgrids = get_filepaths(os.path.join(speaker_folder, "textgrid"))
    transcripts = get_filepaths(os.path.join(speaker_folder, "transcript"))

    assert len(wavs) == len(textgrids) == len(transcripts)

    speaker_name = get_basename(speaker_folder)
    speaker_gender, speaker_accent = speakers_dict[speaker_name]
    accent_name = f"{speaker_accent}-{speaker_name}"
    gender = Gender.MALE if speaker_gender == "M" else Gender.FEMALE

    for wav, textgrid, transcript in zip(wavs, textgrids, transcripts):
      text_en = read_text(transcript)
      final_text = f"{text_en}."
      symbols = text_to_symbols(text_en, lang)

      entry = PreData(
        name=get_basename(wav),
        speaker_name=speaker_name,
        text=final_text,
        wav_path=wav,
        symbols=symbols,
        accents=[accent_name] * len(symbols),
        gender=gender,
        lang=lang
      )

      entries.append(entry)

  entries.sort(key=sort_l2arctic, reverse=False)
  logger.info(f"Parsed {len(entries)} entries from {len(speakers_dict)} speakers.")

  return entries


def sort_l2arctic(entry: PreData) -> str:
  return entry.speaker_name, entry.name


if __name__ == "__main__":
  dest = '/datasets/l2arctic'
  # download(
  #   dir_path = dest
  # )

  res = parse(
    dir_path=dest
  )
