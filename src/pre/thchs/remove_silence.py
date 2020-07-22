import os
from pathlib import Path
from shutil import copyfile

import librosa
import numpy as np
import scipy.signal as sps
from scipy.io import wavfile
from tqdm import tqdm

from src.parser.thchs_kaldi_parser import parse as kaldi_parse
from src.parser.thchs_parser import parse as parse
from src.tacotron.synthesize import to_wav
from src.common.utils import create_parent_folder
from src.common.audio.remove_silence import remove_silence

def remove_silence_thchs(
    origin: str,
    dest: str,
    kaldi_version: bool,
    chunk_size: int,
    threshold_start: float,
    threshold_end: float,
    buffer_start_ms: float,
    buffer_end_ms: float
  ):

  if kaldi_version:
    parsed_data = kaldi_parse(origin)
  else:
    parsed_data = parse(origin)

  print("Removing silence at start and end of wav files...")
  for data in tqdm(parsed_data):
    wav_path = data[3]
    #if speaker_name != 'A11':
    #  continue

    dest_wav_path = wav_path.replace(origin, dest)
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
      b = sent_file.replace(origin, dest)
      copyfile(a, b)
  else:
    a = os.path.join(origin, 'doc/trans/test.word.txt')
    b = os.path.join(dest, 'doc/trans/test.word.txt')
    create_parent_folder(b)
    copyfile(a, b)

    a = os.path.join(origin, 'doc/trans/train.word.txt')
    b = os.path.join(dest, 'doc/trans/train.word.txt')
    copyfile(a, b)
