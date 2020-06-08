import os

def __parse_dataset__(syll_path, wavs_dir):
  with open(syll_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    res = [x.strip() for x in lines]

  files = []
  for x in res:
    pos = x.find(' ')
    name, pinyin = x[:pos], x[pos + 1:]
    
    dir_name, nr = name.split('_')
    nr = int(nr)
    wav_path = os.path.join(wavs_dir, dir_name, name + '.wav')
    exists = os.path.exists(wav_path)
    if not exists:
      print(wav_path)
      continue
    files.append((nr, dir_name, name, wav_path, pinyin))

  return files

def parse(dir_path: str):

  if not os.path.exists(dir_path):
    print("Directory not found:", dir_path)
    raise Exception()

  train_syll = os.path.join(dir_path, 'doc/trans/train.syllable.txt')
  test_syll = os.path.join(dir_path, 'doc/trans/test.syllable.txt')
  train_wavs = os.path.join(dir_path, 'wav/train/')
  test_wavs = os.path.join(dir_path, 'wav/test/')

  train_set = __parse_dataset__(os.path.join(dir_path, train_syll), train_wavs)
  test_set = __parse_dataset__(os.path.join(dir_path, test_syll), test_wavs)
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
