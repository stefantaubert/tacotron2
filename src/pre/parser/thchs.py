import os

from tqdm import tqdm

from src.common.utils import download_tar
from src.pre.parser.pre_data import to_values

def ensure_downloaded(dir_path: str):
  dir_exists = os.path.exists(dir_path)
  if not dir_exists:
    print("THCHS-30 is not downloaded yet.")
    download_tar("http://data.cslt.org/thchs30/zip/wav.tgz", dir_path)
    download_tar("http://data.cslt.org/thchs30/zip/doc.tgz", dir_path)

def parse(dir_path: str) -> list:
  if not os.path.exists(dir_path):
    print("Directory not found:", dir_path)
    raise Exception()

  train_words = os.path.join(dir_path, 'doc/trans/train.word.txt')
  test_words = os.path.join(dir_path, 'doc/trans/test.word.txt')
  train_wavs = os.path.join(dir_path, 'wav/train/')
  test_wavs = os.path.join(dir_path, 'wav/test/')

  parse_paths = [
    (train_words, train_wavs),
    (test_words, test_wavs)
  ]

  files = []

  print("Parsing files...")
  for words_path, wavs_dir in parse_paths:
    with open(words_path, 'r', encoding='utf-8') as f:
      lines = f.readlines()
      res = [x.strip() for x in lines]

    for x in tqdm(res):
      pos = x.find(' ')
      name, chinese = x[:pos], x[pos + 1:]
      
      speaker_name, nr = name.split('_')
      nr = int(nr)
      speaker_name_letter = speaker_name[0]
      speaker_name_number = int(speaker_name[1:])
      wav_path = os.path.join(wavs_dir, speaker_name, name + '.wav')
      exists = os.path.exists(wav_path)
      if not exists:
        print("Not found wav file:", wav_path)
        continue

      # remove "=" from chinese transcription because it is not correct 
      # occurs only in sentences with nr. 374, e.g. B22_374
      chinese = chinese.replace("= ", '')
      files.append((name, speaker_name, chinese, wav_path, speaker_name_letter, speaker_name_number, nr))

  # sort after wav_path
  files.sort(key=lambda tup: (tup[4], tup[5], tup[6]), reverse=False)
  files = [to_values(name=x[0], speaker_name=x[1], text=x[2], wav_path=x[3]) for x in files]
  return files

if __name__ == "__main__":
  dest = '/datasets/thchs_wav'
  ensure_downloaded(
    dir_path = dest
  )

  res = parse(
    dir_path = dest
  )
