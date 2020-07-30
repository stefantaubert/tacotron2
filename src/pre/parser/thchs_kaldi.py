import glob
import os
import shutil
import tempfile

from tqdm import tqdm

from src.common.utils import create_parent_folder, download_tar

def init_download_parser(parser):
  parser.add_argument('--data_dir', type=str, help='THCHS (Kaldi version) dataset directory', required=True)
  return __ensure_downloaded

def __ensure_downloaded(dir_path: str):
  is_downloaded = exists(dir_path)
  if not is_downloaded:
    print("THCHS-30 is not downloaded yet.")
    download_url_kaldi = "http://www.openslr.org/resources/18/data_thchs30.tgz"
    tmp_dir = tempfile.mkdtemp()
    download_tar(download_url_kaldi, tmp_dir)
    subfolder_name = "data_thchs30"
    content_dir = os.path.join(tmp_dir, subfolder_name)
    parent = create_parent_folder(dir_path)
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

  wavs_sents = sorted(tuple(zip(wav_files, sent_files_gen)))
  skipped = [x for x in wavs_sents if x[1] not in sent_files]
  wavs_sents = [x for x in wavs_sents if x[1] in sent_files]
  
  print("Skipped:", len(skipped), "of", len(wavs_sents))
  #print(skipped)

  res = []
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
    #res.append((nr, speaker, basename, wav, chn, sent_file))
    res.append((basename, speaker, chn, wav))
  print("Done.")

  return res

if __name__ == "__main__":
  __ensure_downloaded(
    dir_path = '/datasets/THCHS-30-test'
  )

  res = parse(
    dir_path = '/datasets/THCHS-30'
  )
