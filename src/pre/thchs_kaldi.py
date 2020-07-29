import glob
import os
import shutil
import tempfile
from pathlib import Path
from shutil import copyfile

from tqdm import tqdm

from src.common.audio.remove_silence import remove_silence
from src.common.audio.utils import float_to_wav, upsample
from src.common.utils import create_parent_folder, download_tar
from src.pre.calc_mels import calc_mels


def ensure_upsampled(data_src_dir, data_dest_dir, new_rate=22050):
  already_converted = exists(data_dest_dir)

  if already_converted:
    print("Dataset is already upsampled.")
    return
  
  parsed_data = parse(data_src_dir)

  for data in tqdm(parsed_data):
    wav_path = data[3]

    dest_wav_path = wav_path.replace(data_src_dir, data_dest_dir)
    create_parent_folder(dest_wav_path)

    upsample(wav_path, dest_wav_path, new_rate)

  if kaldi_version:
    for data in tqdm(parsed_data):
      sent_file = data[5]
      a = sent_file
      b = sent_file.replace(data_src_dir, data_dest_dir)
      copyfile(a, b)
  else:
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

  already_removed = exists(data_dest_dir)
 
  if already_removed:
    print("Dataset is already without silence.")
    return
  else:
    print("Saving to {}".format(data_dest_dir))
    
  parsed_data = parse(data_src_dir)

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

  for data in tqdm(parsed_data):
    sent_file = data[5]
    a = sent_file
    b = sent_file.replace(data_src_dir, data_dest_dir)
    copyfile(a, b)

  print("Finished.")

def ensure_downloaded(dir_path: str):
  is_downloaded = exists(dir_path)
  if not is_downloaded:
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

def calc_mels(base_dir: str, name: str, path: str, hparams: str):
  data = parse(path)
  calc_mels(base_dir, name, data, custom_hparams=hparams)


if __name__ == "__main__":
  ensure_downloaded(
    dir_path = '/datasets/THCHS-30-test'
  )

  res = parse(
    dir_path = '/datasets/THCHS-30'
  )

  ensure_upsampled(
    data_src_dir='/datasets/THCHS-30-test',
    data_dest_dir='/datasets/THCHS-30-test-22050',
    new_rate=22050
  )

  remove_silence_main(
    data_src_dir='/datasets/THCHS-30-test-22050',
    data_dest_dir='/datasets/THCHS-30-test_nosil',
    chunk_size=5,
    threshold_start=-25,
    threshold_end=-35,
    buffer_start_ms=100,
    buffer_end_ms=150
  )

  calc_mels(
    base_dir="/datasets/models/taco2pt_v2",
    name="thchs_kaldi",
    path="/datasets/THCHS-30",
    hparams="segment_length=0",
  )
