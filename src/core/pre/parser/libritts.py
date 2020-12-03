import os
from logging import Logger, getLogger
from typing import List, Tuple

from src.core.common.gender import Gender
from src.core.common.language import Language
from src.core.common.utils import (download_tar, get_basename, get_filenames,
                                   get_filepaths, get_subfolders, read_lines,
                                   read_text)
from src.core.pre.parser.data import PreData, PreDataList
from text_utils.text import text_to_symbols
from tqdm import tqdm

# found some invalid text at:
# train-clean-360/8635/295759/8635_295759_000008_000001.original.txt
# "With them were White Thun-der, who had charge of the "speech-belts," and Sil-ver Heels, who was swift of foot."


def download(dir_path: str):
  pass


def parse(dir_path: str, logger: Logger = getLogger()) -> PreDataList:
  if not os.path.exists(dir_path):
    ex = ValueError(f"Directory not found: {dir_path}")
    logger.error("", exc_info=ex)
    raise ex

  speakers_path = os.path.join(dir_path, "SPEAKERS.txt")
  speakers = read_lines(speakers_path)
  speakers = speakers[12:]
  speakers_dict = {}
  for speaker_details in speakers:
    s_id, gender, _, _, name = speaker_details.split(" | ")
    speakers_dict[s_id.strip()] = name.strip(), gender.strip()

  lang = Language.ENG

  entries = PreDataList()

  logger.info("Parsing files...")
  for dataset_folder in tqdm(get_subfolders(dir_path)):
    logger.info(f"Parsing {get_basename(dataset_folder)}...")

    for speaker_folder in tqdm(get_subfolders(dataset_folder)):
      speaker_id = get_basename(speaker_folder)
      speaker_name, speaker_gender = speakers_dict[speaker_id]
      accent_name = speaker_name
      gender = Gender.MALE if speaker_gender == "M" else Gender.FEMALE

      for chapter_folder in get_subfolders(speaker_folder):
        files = get_filepaths(chapter_folder)
        wavs = [x for x in files if x.endswith(".wav")]
        texts = [x for x in files if x.endswith(".normalized.txt")]
        assert len(wavs) == len(texts)

        for wav_file, text_file in zip(wavs, texts):
          assert get_basename(wav_file) == get_basename(text_file)[:-len(".normalized")]
          text_en = read_text(text_file)
          symbols = text_to_symbols(
            text=text_en,
            lang=lang,
            ipa_settings=None,
            logger=logger,
          )

          entry = PreData(
            name=get_basename(wav_file),
            speaker_name=speaker_name,
            text=text_en,
            wav_path=wav_file,
            symbols=symbols,
            accents=[accent_name] * len(symbols),
            gender=gender,
            lang=lang
          )

          entries.append(entry)

  entries.sort(key=sort_libri, reverse=False)
  logger.info(f"Parsed {len(entries)} entries from {len(speakers_dict)} speakers.")

  return entries


def sort_libri(entry: PreData) -> str:
  return entry.speaker_name, entry.name


if __name__ == "__main__":
  dest = '/datasets/libriTTS'
  # download(
  #   dir_path = dest
  # )

  res = parse(
    dir_path=dest
  )
