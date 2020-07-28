from src.parser.thchs_parser import parse, exists
from src.parser.thchs_kaldi_parser import parse as kaldi_parse, exists as kaldi_exists
from shutil import copyfile
from tqdm import tqdm
import librosa    
from scipy.io import wavfile
import scipy.signal as sps
import os
from pathlib import Path
import numpy as np
from src.common.audio.utils import float_to_wav
from src.common.utils import create_parent_folder

def upsample(origin, dest, new_rate):
  new_data, _ = librosa.load(origin, sr=new_rate, mono=True, dtype=np.float32)
  float_to_wav(wav=new_data, path=dest, sample_rate=new_rate, normalize=False)

def ensure_upsampled(data_src_dir, data_dest_dir, kaldi_version: bool, new_rate=22050):
  if kaldi_version:
    already_converted = kaldi_exists(data_dest_dir)
  else:
    already_converted = exists(data_dest_dir)

  if already_converted:
    print("Dataset is already upsampled.")
    return
  
  if kaldi_version:
    parsed_data = kaldi_parse(data_src_dir)
  else:
    parsed_data = parse(data_src_dir)

  for data in tqdm(parsed_data):
    wav_path = data[3]
    #if speaker_name != 'A11':
    #  continue

    dest_wav_path = wav_path.replace(data_src_dir, data_dest_dir)
    create_parent_folder(dest_wav_path)

    upsample(wav_path, dest_wav_path, new_rate)

    #new_data = new_data.astype(np.uint16)
    #new_data = (new_data * 32767).astype(np.int16)
    #wavfile.write(dest_wav_path, new_rate, new_data)

    #new_data = ints.astype('<u2')
    #new_data = little_endian.tostring()
    #sf.write(tmp_file, audio, rate, subtype='PCM_16')
    #librosa.output.write_wav(dest_wav_path, new_data, new_rate)

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

if __name__ == "__main__":
  ensure_upsampled(
    data_src_dir='/datasets/thchs_wav',
    data_dest_dir='/datasets/thchs_16bit_22050kHz',
    kaldi_version=False,
    new_rate=22050
  )