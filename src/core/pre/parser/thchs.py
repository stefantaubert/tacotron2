import os
from src.core.common.language import Language
from src.core.common.text import text_to_symbols
from src.core.common.utils import download_tar

from tqdm import tqdm

from src.core.pre.parser.data import PreDataList, PreData
from typing import List, Tuple

__question_particle1 = '吗'
__question_particle2 = '呢'

def download(dir_path: str):
  download_tar("http://data.cslt.org/thchs30/zip/wav.tgz", dir_path)
  download_tar("http://data.cslt.org/thchs30/zip/doc.tgz", dir_path)

def parse(dir_path: str) -> PreDataList:
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

  files: List[Tuple[Tuple[str, int, int], PreData]] = []

  print("Parsing files...")
  for words_path, wavs_dir in parse_paths:
    with open(words_path, 'r', encoding='utf-8') as f:
      lines = f.readlines()
      res = [x.strip() for x in lines]

    for x in tqdm(res):
      pos = x.find(' ')
      name, chinese = x[:pos], x[pos + 1:]

      speaker_name, nr = name.split("_")
      nr = int(nr)
      speaker_name_letter = speaker_name[0]
      speaker_name_number = int(speaker_name[1:])
      wav_path = os.path.join(wavs_dir, speaker_name, name + '.wav')
      exists = os.path.exists(wav_path)
      if not exists:
        wav_path = os.path.join(wavs_dir, speaker_name, name + '.WAV')
      exists = os.path.exists(wav_path)
      if not exists:
        print("Not found wav file:", wav_path)
        continue

      # remove "=" from chinese transcription because it is not correct
      # occurs only in sentences with nr. 374, e.g. B22_374
      chinese = chinese.replace("= ", '')
      is_question = str.endswith(chinese, __question_particle1) or str.endswith(chinese, __question_particle2)
      if is_question:
        chinese += "？"
      else:
        chinese += "。"

      symbols = text_to_symbols(chinese, Language.CHN)
      accents = [speaker_name] * len(symbols)
      tmp = PreData(name, speaker_name, chinese, wav_path, symbols, accents, Language.CHN)

      files.append((tmp, (speaker_name_letter, speaker_name_number, nr)))

  files.sort(key=lambda tup: tup[1], reverse=False)
  res = PreDataList([x for x, _ in files])
  return res

if __name__ == "__main__":
  dest = '/datasets/thchs_wav'
  # download(
  #   dir_path = dest
  # )

  res = parse(
    dir_path = dest
  )
