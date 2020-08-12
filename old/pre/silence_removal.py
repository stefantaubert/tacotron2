import os
from shutil import copyfile

import matplotlib.pylab as plt
from tqdm import tqdm

from src.core.common.audio.utils import remove_silence_file, wav_to_float32
from src.core.common.utils import stack_images_vertically
from src.pre.mel_parser import MelParser, plot_melspec
from src.pre.wav_pre_io import parse_data, save_data, already_exists, WavData, WavDataList, get_wavs_dir
from src.tacotron.hparams import create_hparams


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

  remove_silence_file(
    in_path = wav_orig,
    out_path = wav_trimmed,
    chunk_size = chunk_size,
    threshold_start = threshold_start,
    threshold_end = threshold_end,
    buffer_start_ms = buffer_start_ms,
    buffer_end_ms = buffer_end_ms
  )
  
  hparams = create_hparams()
  plotter = MelParser(hparams)

  mel = plotter.get_mel_tensor_from_file(wav_orig)
  ax = plot_melspec(mel, title="Original")
  a_out = os.path.join(dest_dir, basename + ".png")
  plt.savefig(a_out, bbox_inches='tight')

  mel = plotter.get_mel_tensor_from_file(wav_trimmed)
  ax = plot_melspec(mel, title="Trimmed")
  b_out = os.path.join(dest_dir, dest_name + ".png")
  plt.savefig(b_out, bbox_inches='tight')

  comp_out = os.path.join(dest_dir, dest_name + "_comp.png")
  stack_images_vertically([a_out, b_out], comp_out)
  print("Saved results to: {}".format(dest_dir))

def __remove_silence(
  base_dir: str,
  source_name: str,
  destination_name: str,
  chunk_size: int,
  threshold_start: float,
  threshold_end: float,
  buffer_start_ms: float,
  buffer_end_ms: float):
  
  if not already_exists(base_dir, destination_name):
    data = parse_data(base_dir, source_name)
    dest_dir = get_wavs_dir(base_dir, destination_name)
    result: WavDataList = []

    removed_silence_duration = 0
    print("Removing silence...")
    values: WavData
    for values in tqdm(data):
      dest_wav_path = os.path.join(dest_dir, "{}_{}.wav".format(values.i, values.basename))
      
      new_duration = remove_silence_file(
        in_path = values.wav,
        out_path = dest_wav_path,
        chunk_size = chunk_size,
        threshold_start = threshold_start,
        threshold_end = threshold_end,
        buffer_start_ms = buffer_start_ms,
        buffer_end_ms = buffer_end_ms
      )

      removed_silence_duration += values.duration - new_duration
      values.duration = new_duration
      values.wav = dest_wav_path
      result.append(values)

    save_data(base_dir, destination_name, result)
    print("Removed {}m of silence.".format(removed_silence_duration / 60))

def init_remove_silence_parser(parser):
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--source_name', type=str, help='THCHS dataset directory', required=True)
  parser.add_argument('--destination_name', type=str, help='THCHS destination directory', required=True)
  parser.add_argument('--chunk_size', type=int, required=True)
  parser.add_argument('--threshold_start', type=float, required=True)
  parser.add_argument('--threshold_end', type=float, required=True)
  parser.add_argument('--buffer_start_ms', type=float, help="amount of factors of chunk_size at the beginning and the end should be reserved", required=True)
  parser.add_argument('--buffer_end_ms', type=float, help="amount of factors of chunk_size at the beginning and the end should be reserved", required=True)
  return __remove_silence

if __name__ == "__main__":
  # #in_path = "/datasets/models/taco2pt_v2/analysis/trimming/old version/A13_224/A13_224.wav"
  # #in_path = "/datasets/models/taco2pt_v2/wavs/thchs_22050kHz_normalized/12202_D31_956.wav"
  # in_path = "/datasets/models/taco2pt_v2/wavs/thchs_22050kHz_normalized/5799_B32_251.wav"
  # #in_path = "/datasets/models/taco2pt_v2/wavs/thchs_22050kHz_normalized/5799_B32_251.wav"

  # __remove_silence_plot(
  #   in_path = in_path,
  #   base_dir = "/datasets/models/taco2pt_v2/analysis/trimming",
  #   threshold_start = -20,
  #   threshold_end = -30,
  #   chunk_size = 5,
  #   buffer_start_ms = 100,
  #   buffer_end_ms = 150
  # )

  __remove_silence(
    base_dir="/datasets/models/taco2pt_v2",
    source_name='thchs_22050kHz_normalized',
    destination_name='thchs_22050kHz_normalized_nosil',
    threshold_start = -20,
    threshold_end = -30,
    chunk_size = 5,
    buffer_start_ms = 100,
    buffer_end_ms = 150
  )

  # __remove_silence(
  #   base_dir="/datasets/models/taco2pt_v2",
  #   source_name='thchs_kaldi_22050kHz',
  #   destination_name='thchs_kaldi_22050kHz_nosil',
  #   chunk_size=5,
  #   threshold_start=-25,
  #   threshold_end=-35,
  #   buffer_start_ms=100,
  #   buffer_end_ms=150
  # )
