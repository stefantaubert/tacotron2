import os
from utils import download_tar
import tempfile
import shutil
from pathlib import Path

def ensure_downloaded(dir_path: str):
  is_downloaded = exists(dir_path)
  if not is_downloaded:
    __download_dataset(dir_path)

def __download_dataset(dir_path: str):
  print("THCHS-30 is not downloaded yet.")
  # old ones:
  # - http://data.cslt.org/thchs30/zip/wav.tgz
  # - http://data.cslt.org/thchs30/zip/doc.tgz
  download_url_kaldi = "http://www.openslr.org/resources/18/data_thchs30.tgz"
  tmp_dir = tempfile.mkdtemp()
  download_tar(download_url_kaldi, tmp_dir)
  subfolder_name = "data_thchs30"
  content_dir = os.path.join(tmp_dir, subfolder_name)
  parent = Path(dir_path).parent
  os.makedirs(parent, exist_ok=True)
  dest = os.path.join(parent, subfolder_name)
  shutil.move(content_dir, dest)
  os.rename(dest, dir_path)

def exists(dir_path: str):
  path_to_check = os.path.join(dir_path, 'doc/trans/train.word.txt')
  result = os.path.exists(path_to_check)
  return result

def __parse_dataset__(words_path, wavs_dir):
  with open(words_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    res = [x.strip() for x in lines]

  files = []
  for x in res:
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

    files.append((nr, speaker_name, name, wav_path, chinese))

  return files

def parse(dir_path: str):

  if not os.path.exists(dir_path):
    print("Directory not found:", dir_path)
    raise Exception()

  train_words = os.path.join(dir_path, 'doc/trans/train.word.txt')
  test_words = os.path.join(dir_path, 'doc/trans/test.word.txt')
  train_wavs = os.path.join(dir_path, 'wav/train/')
  test_wavs = os.path.join(dir_path, 'wav/test/')

  train_set = __parse_dataset__(os.path.join(dir_path, train_words), train_wavs)
  test_set = __parse_dataset__(os.path.join(dir_path, test_words), test_wavs)
  #print(train_set[0:10])
  #print(test_set[0:10])

  res = []
  res.extend(train_set)
  res.extend(test_set)
  res.sort()


  return res

if __name__ == "__main__":
  dir_path = '/datasets/thchs_wav'
  res = parse(dir_path)
  print(res[:10])
