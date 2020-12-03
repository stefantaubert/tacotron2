import os
import shutil
import tarfile
from logging import Logger, getLogger

import wget
from src.core.common.gender import Gender
from src.core.common.language import Language
from src.core.common.utils import read_lines
from src.core.pre.parser.data import PreData, PreDataList
from text_utils.text import text_to_symbols
from tqdm import tqdm


def download(dir_path: str):
  print("LJSpeech is not downloaded yet.")
  print("Starting download...")
  os.makedirs(dir_path, exist_ok=False)
  download_url = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
  dest = wget.download(download_url, dir_path)
  downloaded_file = os.path.join(dir_path, dest)
  print("\nFinished download to {}".format(downloaded_file))
  print("Unpacking...")
  tar = tarfile.open(downloaded_file, "r:bz2")
  tar.extractall(dir_path)
  tar.close()
  print("Done.")
  print("Moving files...")
  dir_name = "LJSpeech-1.1"
  ljs_data_dir = os.path.join(dir_path, dir_name)
  files = os.listdir(ljs_data_dir)
  for f in tqdm(files):
    shutil.move(os.path.join(ljs_data_dir, f), dir_path)
  print("Done.")
  os.remove(downloaded_file)
  os.rmdir(ljs_data_dir)


def parse(dir_path: str, logger: Logger = getLogger()) -> PreDataList:
  if not os.path.exists(dir_path):
    ex = ValueError(f"Directory not found: {dir_path}")
    logger.error("", exc_info=ex)
    raise ex

  metadata_filepath = os.path.join(dir_path, 'metadata.csv')

  if not os.path.exists(metadata_filepath):
    ex = ValueError(f"Metadatafile not found: {metadata_filepath}")
    logger.error("", exc_info=ex)
    raise ex

  wav_dirpath = os.path.join(dir_path, 'wavs')

  if not os.path.exists(wav_dirpath):
    ex = ValueError(f"WAVs not found: {wav_dirpath}")
    logger.error("", exc_info=ex)
    raise ex

  result = PreDataList()
  speaker_name = '1'
  accent_name = "north_america"
  lang = Language.ENG
  gender = Gender.FEMALE

  lines = read_lines(metadata_filepath)
  logger.info("Parsing files...")
  for line in tqdm(lines):
    parts = line.split('|')
    basename = parts[0]
    # parts[1] contains years, in parts[2] the years are written out
    # ex. ['LJ001-0045', '1469, 1470;', 'fourteen sixty-nine, fourteen seventy;']
    wav_path = os.path.join(wav_dirpath, f'{basename}.wav')
    text = parts[2]
    symbols = text_to_symbols(
      text=text,
      lang=lang,
      ipa_settings=None,
      logger=logger,
    )

    entry = PreData(
      name=basename,
      speaker_name=speaker_name,
      text=text,
      wav_path=wav_path,
      symbols=symbols,
      accents=[accent_name] * len(symbols),
      gender=gender,
      lang=lang
    )

    result.append(entry)

  result.sort(key=sort_ljs, reverse=False)
  print("Done.")

  return result


def sort_ljs(entry: PreData) -> str:
  return entry.name


if __name__ == "__main__":
  # download(
  #   dir_path = '/datasets/LJSpeech-1.1'
  # )

  tmp = parse(
    dir_path='/datasets/LJSpeech-1.1'
  )
