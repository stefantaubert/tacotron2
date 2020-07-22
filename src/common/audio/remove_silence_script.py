import argparse
from src.pre.audio.remove_silence import remove_silence
from src.waveglow.hparams import create_hparams
from src.tacotron.script_plot_mel import (Mel2Samp, get_audio, get_segment, plot_melspec, stack_images_vertically)
import matplotlib.pylab as plt
from shutil import copyfile
import os


def remove_silence_plot(
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
  parser = argparse.ArgumentParser()
  parser.add_argument('--no_debugging', action='store_true')
  parser.add_argument('--base_dir', type=str, help='base directory')
  parser.add_argument('--wav', type=str)
  parser.add_argument('--chunk_size', type=int)
  parser.add_argument('--threshold_start', type=float)
  parser.add_argument('--threshold_end', type=float)
  parser.add_argument('--buffer_start_ms', type=float, help="amount of factors of chunk_size at the beginning and the end should be reserved")
  parser.add_argument('--buffer_end_ms', type=float, help="amount of factors of chunk_size at the beginning and the end should be reserved")

  args = parser.parse_args()

  if not args.no_debugging:
    args.base_dir = "/datasets/models/taco2pt_v2/analysis/trimming"
    args.wav = "/datasets/thchs_16bit_22050kHz/wav/train/A13/A13_224.wav"
    args.threshold_start = -25
    args.threshold_end = -25
    args.chunk_size = 5
    args.buffer_start_ms = 25
    args.buffer_end_ms = 25

  remove_silence_plot(
    in_path=args.wav,
    base_dir=args.base_dir,
    chunk_size = args.chunk_size,
    threshold_start = args.threshold_start,
    threshold_end = args.threshold_end,
    buffer_start_ms = args.buffer_start_ms,
    buffer_end_ms = args.buffer_end_ms
  )
