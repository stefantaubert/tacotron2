import os
import shutil
import tarfile

import wget
from tqdm import tqdm

from src.core.pre.language import Language
from src.core.pre.parser.data import PreDataList, PreData

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

def parse(path: str) -> PreDataList:
  if not os.path.exists(path):
    print("Directory not found:", path)
    raise Exception()

  metadata_filepath = os.path.join(path, 'metadata.csv')

  if not os.path.exists(metadata_filepath):
    print("Metadatafile not found:", metadata_filepath)
    raise Exception()

  wav_dirpath = os.path.join(path, 'wavs')

  if not os.path.exists(wav_dirpath):
    print("WAVs not found:", wav_dirpath)
    raise Exception()

  index = 1
  result = []
  speaker_name = '1'

  with open(metadata_filepath, encoding='utf-8') as f:
    lines = f.readlines()
  
  print("Parsing files...")
  for line in tqdm(lines):
    parts = line.strip().split('|')
    basename = parts[0]
    # parts[1] contains years, in parts[2] the years are written out
    # ex. ['LJ001-0045', '1469, 1470;', 'fourteen sixty-nine, fourteen seventy;']
    wav_path = os.path.join(wav_dirpath, '{}.wav'.format(basename))
    text = parts[2]
    tmp = (basename, speaker_name, text, wav_path)
    result.append(tmp)
  print("Done.")

  # sort after basename
  result.sort(key=lambda tup: tup[0], reverse=False)
  result = PreDataList([PreData(x[0], x[1], x[2], x[3], Language.ENG) for i, x in enumerate(result)])

  return result

if __name__ == "__main__":
  download(
    dir_path = '/datasets/LJSpeech-1.1'
  )

  result = parse(
    path = '/datasets/LJSpeech-1.1'
  )
