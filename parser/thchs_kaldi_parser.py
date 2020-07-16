import os
#from utils import download_tar
import tempfile
import shutil
import glob
from pathlib import Path
from tqdm import tqdm

def ensure_downloaded(dir_path: str):
  is_downloaded = exists(dir_path)
  if not is_downloaded:
    __download_dataset(dir_path)

def __download_dataset(dir_path: str):
  print("THCHS-30 is not downloaded yet.")
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
  path_to_check = os.path.join(dir_path, "data", 'D32_999.wav.trn')
  result = os.path.exists(path_to_check)
  return result

def parse(dir_path: str):

  if not os.path.exists(dir_path):
    print("Directory not found:", dir_path)
    raise Exception()
  
  sent_paths = os.path.join(dir_path, "data", "*.trn")
  wav_paths = os.path.join(dir_path, "data", "*.wav")
  sent_files = glob.glob(sent_paths)
  wav_files = glob.glob(wav_paths)
  sent_files_gen = ["{}.trn".format(x) for x in wav_files]

  for gen_sent_file in sent_files_gen:
    if gen_sent_file not in sent_files:
      raise Exception

  wavs_sents = tuple(zip(wav_files, sent_files_gen))

  res = []
  wavs_sents = sorted(wavs_sents)
  print("Parsing files...")
  for wav, sent_file in tqdm(wavs_sents):
    with open(sent_file, 'r', encoding='utf-8') as f:
      content = f.readlines()
    chn = content[0].strip()
    # remove "=" from chinese transcription because it is not correct 
    # occurs only in sentences with nr. 374, e.g. B22_374
    chn = chn.replace("= ", '')
    basename = os.path.basename(wav)[:-4]
    speaker, nr = basename.split('_')
    nr = int(nr)
    res.append((nr, speaker, basename, wav, chn, sent_file))

  return res

if __name__ == "__main__":
  dir_path = '/datasets/phil_home/datasets/THCHS-30'
  res = parse(dir_path)
  print(res[:10])
