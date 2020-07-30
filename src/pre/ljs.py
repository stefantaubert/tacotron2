import os
import shutil
import tarfile

import wget
from tqdm import tqdm
from src.pre.calc_mels import calc_mels
from src.pre.text_pre import preprocess

def init_download_parser(parser):
  parser.add_argument('--dir_path', type=str, help='LJS dataset directory', required=True)
  return __ensure_downloaded

def __ensure_downloaded(dir_path: str):
  metadata_filepath = __get_metadata_filepath(dir_path)
  metadata_file_exists = os.path.exists(metadata_filepath)
  if not metadata_file_exists:
    print("LJSpeech is not downloaded yet.")
    print("Starting download...")
    os.makedirs(dir_path, exist_ok=True)
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

def __get_metadata_filepath(root_dir: str) -> str:
  result = os.path.join(root_dir, 'metadata.csv')
  return result

def __get_wav_dirpath(root_dir) -> str:
  result = os.path.join(root_dir, 'wavs')
  return result

def __parse(path: str):
  if not os.path.exists(path):
    print("Directory not found:", path)
    raise Exception()

  metadata_filepath = __get_metadata_filepath(path)

  if not os.path.exists(metadata_filepath):
    print("Metadatafile not found:", metadata_filepath)
    raise Exception()

  wav_dirpath = __get_wav_dirpath(path)

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

  return result

def init_calc_mels_parser(parser):
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--name', type=str, required=True)
  parser.add_argument('--path', type=str, required=True)
  parser.add_argument('--hparams', type=str)
  return __calc_mels

def __calc_mels(base_dir: str, name: str, path: str, hparams: str):
  data = __parse(path)
  __calc_mels(base_dir, name, data, custom_hparams=hparams)

def init_text_pre_parser(parser):
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--mel_name', type=str, required=True)
  parser.add_argument('--ds_name', type=str, help='the name you want to call the dataset', required=True)
  parser.add_argument('--convert_to_ipa', action='store_true', help='transcribe to IPA')
  parser.set_defaults(ignore_tones=True, ignore_arcs=True, lang="eng")
  return preprocess

if __name__ == "__main__":

  __ensure_downloaded(
    data_dir = '/datasets/LJSpeech-1.1-tmp'
  )

  result = __parse(
    path = '/datasets/LJSpeech-1.1-test'
  )

  __calc_mels(
    base_dir="/datasets/models/taco2pt_v2",
    name="ljs",
    path="/datasets/LJSpeech-1.1",
    hparams="segment_length=0",
  )