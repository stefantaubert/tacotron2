import argparse
import os
from shutil import copyfile

import matplotlib.pylab as plt
from pydub import AudioSegment

from src.tacotron.plot_mel import (plot_melspec, stack_images_vertically)
from src.waveglow.hparams import create_hparams


def detect_leading_silence(sound, silence_threshold=-50.0, chunk_size=10, buffer_ms: int = 0):
    '''
    sound is a pydub.AudioSegment
    silence_threshold in dB
    chunk_size in ms

    iterate over chunks until you find the first one with sound
    '''
    assert chunk_size > 0

    trim_ms = 0

    while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold and trim_ms < len(sound):
      trim_ms += chunk_size

    if buffer_ms <= trim_ms:
      trim_ms -= buffer_ms

    return trim_ms

def remove_silence(
    in_path: str,
    out_path: str,
    chunk_size: int,
    threshold_start: float,
    threshold_end: float,
    buffer_start_ms: float,
    buffer_end_ms: float
  ):

  sound = AudioSegment.from_file(in_path, format="wav")

  start_trim = detect_leading_silence(
    sound=sound,
    silence_threshold=threshold_start,
    chunk_size=chunk_size,
    buffer_ms=buffer_start_ms
  )

  end_trim = detect_leading_silence(
    sound=sound.reverse(),
    silence_threshold=threshold_end,
    chunk_size=chunk_size,
    buffer_ms=buffer_end_ms
  )
  
  duration = len(sound)
  trimmed_sound = sound[start_trim:duration - end_trim]
  trimmed_sound.export(out_path, format="wav")
  
  # sound.duration_seconds is not actualized
  

def init_remove_silence_plot_parser(parser):
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--wav', type=str, required=True)
  parser.add_argument('--chunk_size', type=int, required=True)
  parser.add_argument('--threshold_start', type=float, required=True)
  parser.add_argument('--threshold_end', type=float, required=True)
  parser.add_argument('--buffer_start_ms', type=float, help="amount of factors of chunk_size at the beginning and the end should be reserved", required=True)
  parser.add_argument('--buffer_end_ms', type=float, help="amount of factors of chunk_size at the beginning and the end should be reserved", required=True)
  return __remove_silence_plot

def __remove_silence_plot(
    in_path: str,
    base_dir: str,
    chunk_size: int,
    threshold_start: float,
    threshold_end: float,
    buffer_start_ms: float,
    buffer_end_ms: float
  ):

  basename = os.path.basename(in_path)[:-4]
  dest_dir = os.path.join(base_dir, basename)
  
  os.makedirs(dest_dir, exist_ok=True)

  wav_orig = os.path.join(dest_dir, basename + ".wav")
  copyfile(in_path, wav_orig)
  dest_name = "trimmed_{}_{}dBFS_{}ms_{}dBFS_{}ms".format(chunk_size, threshold_start, buffer_start_ms, threshold_end, buffer_end_ms)
  wav_trimmed = os.path.join(dest_dir, dest_name + ".wav")

  remove_silence(
    in_path = wav_orig,
    out_path = wav_trimmed,
    chunk_size = chunk_size,
    threshold_start = threshold_start,
    threshold_end = threshold_end,
    buffer_start_ms = buffer_start_ms,
    buffer_end_ms = buffer_end_ms
  )
  
  hparams = create_hparams()
  plotter = Mel2Samp(hparams)

  wav_a = get_audio(wav_orig)
  mel = plotter.get_mel(wav_a)
  ax = plot_melspec(mel, title="Original")
  a_out = os.path.join(dest_dir, basename + ".png")
  plt.savefig(a_out, bbox_inches='tight')

  wav_b = get_audio(wav_trimmed)
  mel = plotter.get_mel(wav_b)
  ax = plot_melspec(mel, title="Trimmed")
  b_out = os.path.join(dest_dir, dest_name + ".png")
  plt.savefig(b_out, bbox_inches='tight')

  comp_out = os.path.join(dest_dir, dest_name + "_comp.png")
  stack_images_vertically([a_out, b_out], comp_out)
  print("Saved results to: {}".format(dest_dir))

if __name__ == "__main__":
  import tempfile
  dest = tempfile.mktemp()
  remove_silence(
    in_path = "/datasets/thchs_16bit_22050kHz/wav/train/A13/A13_224.wav",
    out_path = dest,
    threshold_start = -25,
    threshold_end = -25,
    chunk_size = 5,
    buffer_start_ms = 25,
    buffer_end_ms = 25
  )



  __remove_silence_plot(
    base_dir = "/datasets/models/taco2pt_v2/analysis/trimming",
    wav = "/datasets/thchs_16bit_22050kHz/wav/train/A13/A13_224.wav",
    threshold_start = -25,
    threshold_end = -25,
    chunk_size = 5,
    buffer_start_ms = 25,
    buffer_end_ms = 25
  )