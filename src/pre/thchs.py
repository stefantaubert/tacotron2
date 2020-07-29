import os
import shutil
import tempfile
from pathlib import Path
from shutil import copyfile

from scipy.io import wavfile
from tqdm import tqdm

from src.common.audio.remove_silence import remove_silence
from src.common.audio.utils import float_to_wav, upsample
from src.common.utils import create_parent_folder, download_tar
from src.pre.calc_mels import calc_mels


def ensure_upsampled(data_src_dir, data_dest_dir, new_rate=22050):
  already_converted = __exists(data_dest_dir)

  if already_converted:
    print("Dataset is already upsampled.")
    return
  
  parsed_data = __parse(data_src_dir)

  for data in tqdm(parsed_data):
    wav_path = data[3]

    dest_wav_path = wav_path.replace(data_src_dir, data_dest_dir)
    create_parent_folder(dest_wav_path)

    upsample(wav_path, dest_wav_path, new_rate)

    a = os.path.join(data_src_dir, 'doc/trans/test.word.txt')
    b = os.path.join(data_dest_dir, 'doc/trans/test.word.txt')
    create_parent_folder(b)
    copyfile(a, b)

    a = os.path.join(data_src_dir, 'doc/trans/train.word.txt')
    b = os.path.join(data_dest_dir, 'doc/trans/train.word.txt')
    copyfile(a, b)

def remove_silence_main(
    data_src_dir: str,
    data_dest_dir: str,
    chunk_size: int,
    threshold_start: float,
    threshold_end: float,
    buffer_start_ms: float,
    buffer_end_ms: float
  ):

  already_removed = __exists(data_dest_dir)
  
  if already_removed:
    print("Dataset is already without silence.")
    return
  else:
    print("Saving to {}".format(data_dest_dir))
    
  parsed_data = __parse(data_src_dir)

  print("Removing silence at start and end of wav files...")
  for data in tqdm(parsed_data):
    wav_path = data[3]

    dest_wav_path = wav_path.replace(data_src_dir, data_dest_dir)
    create_parent_folder(dest_wav_path)

    remove_silence(
      in_path = wav_path,
      out_path = dest_wav_path,
      chunk_size = chunk_size,
      threshold_start = threshold_start,
      threshold_end = threshold_end,
      buffer_start_ms = buffer_start_ms,
      buffer_end_ms = buffer_end_ms
    )

  a = os.path.join(data_src_dir, 'doc/trans/test.word.txt')
  b = os.path.join(data_dest_dir, 'doc/trans/test.word.txt')
  create_parent_folder(b)
  copyfile(a, b)

  a = os.path.join(data_src_dir, 'doc/trans/train.word.txt')
  b = os.path.join(data_dest_dir, 'doc/trans/train.word.txt')
  copyfile(a, b)
  print("Finished.")

def ensure_downloaded(dir_path: str):
  is_downloaded = __exists(dir_path)
  if not is_downloaded:
    print("THCHS-30 is not downloaded yet.")
    download_tar("http://data.cslt.org/thchs30/zip/wav.tgz", dir_path)
    download_tar("http://data.cslt.org/thchs30/zip/doc.tgz", dir_path)

def __exists(dir_path: str):
  path_to_check = os.path.join(dir_path, 'doc/trans/train.word.txt')
  result = os.path.exists(path_to_check)
  return result

def __parse_dataset__(words_path, wavs_dir):
  with open(words_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    res = [x.strip() for x in lines]

  files = []
  for x in tqdm(res):
    pos = x.find(' ')
    name, chinese = x[:pos], x[pos + 1:]
    
    speaker_name, nr = name.split('_')
    nr = int(nr)
    wav_path = os.path.join(wavs_dir, speaker_name, name + '.wav')
    exists = os.path.exists(wav_path)
    if not exists:
      print("Not found wav file:", wav_path)
      continue

    # remove "=" from chinese transcription because it is not correct 
    # occurs only in sentences with nr. 374, e.g. B22_374
    chinese = chinese.replace("= ", '')
    #files.append((nr, speaker_name, name, wav_path, chinese))
    files.append((name, speaker_name, chinese, wav_path))

  return files

def __parse(dir_path: str):
  if not os.path.exists(dir_path):
    print("Directory not found:", dir_path)
    raise Exception()

  train_words = os.path.join(dir_path, 'doc/trans/train.word.txt')
  test_words = os.path.join(dir_path, 'doc/trans/test.word.txt')
  train_wavs = os.path.join(dir_path, 'wav/train/')
  test_wavs = os.path.join(dir_path, 'wav/test/')

  print("Parsing files...")
  print("Part 1/2...")
  train_set = __parse_dataset__(os.path.join(dir_path, train_words), train_wavs)
  print("Part 2/2...")
  test_set = __parse_dataset__(os.path.join(dir_path, test_words), test_wavs)
  print("Done.")
  #print(train_set[0:10])
  #print(test_set[0:10])

  res = []
  res.extend(train_set)
  res.extend(test_set)
  res.sort()

  return res

def calc_mels(base_dir: str, name: str, path: str, hparams: str):
  data = __parse(path)
  calc_mels(base_dir, name, data, custom_hparams=hparams)

if __name__ == "__main__":
  ensure_downloaded(
    dir_path = '/datasets/thchs_wav'
  )

  res = __parse(
    dir_path='/datasets/thchs_wav'
  )

  remove_silence_main(
    data_src_dir='/datasets/thchs_16bit_22050kHz',
    data_dest_dir='/datasets/thchs_16bit_22050kHz_nosil',
    kaldi_version=False,
    chunk_size=5,
    threshold_start=-25,
    threshold_end=-35,
    buffer_start_ms=100,
    buffer_end_ms=150
  )

  ensure_upsampled(
    data_src_dir='/datasets/thchs_wav',
    data_dest_dir='/datasets/thchs_16bit_22050kHz',
    new_rate=22050
  )

  calc_mels(
    base_dir="/datasets/models/taco2pt_v2",
    name="thchs",
    path="/datasets/thchs_16bit_22050kHz_nosil",
    hparams="segment_length=0",
  )
  
