import os

from tqdm import tqdm

from src.paths import get_wavs_dir
from src.pre.wav_pre import parse_data, save_data
from src.common.audio.remove_silence import remove_silence


def __remove_silence(
  base_dir: str,
  source_name: str,
  destination_name: str,
  chunk_size: int,
  threshold_start: float,
  threshold_end: float,
  buffer_start_ms: float,
  buffer_end_ms: float):
  data = parse_data(base_dir, source_name)
  dest_dir = get_wavs_dir(base_dir, destination_name)
  result = []

  print("Removing silence...")
  for i, values in tqdm(enumerate(data), total=len(data)):
    dest_wav_path = os.path.join(dest_dir, "{}.wav".format(i))
    wav_path = values[3]
    
    remove_silence(
      in_path = wav_path,
      out_path = dest_wav_path,
      chunk_size = chunk_size,
      threshold_start = threshold_start,
      threshold_end = threshold_end,
      buffer_start_ms = buffer_start_ms,
      buffer_end_ms = buffer_end_ms
    )

    values[3] = dest_wav_path
    result.append(values)
  save_data(base_dir, destination_name, result)

def init_remove_silence_parser(parser):
  parser.add_argument('--data_src_dir', type=str, help='THCHS dataset directory', required=True)
  parser.add_argument('--data_dest_dir', type=str, help='THCHS destination directory', required=True)
  parser.add_argument('--chunk_size', type=int, required=True)
  parser.add_argument('--threshold_start', type=float, required=True)
  parser.add_argument('--threshold_end', type=float, required=True)
  parser.add_argument('--buffer_start_ms', type=float, help="amount of factors of chunk_size at the beginning and the end should be reserved", required=True)
  parser.add_argument('--buffer_end_ms', type=float, help="amount of factors of chunk_size at the beginning and the end should be reserved", required=True)
  return __remove_silence

if __name__ == "__main__":
  __remove_silence(
    data_src_dir='/datasets/thchs_16bit_22050kHz',
    data_dest_dir='/datasets/thchs_16bit_22050kHz_nosil',
    kaldi_version=False,
    chunk_size=5,
    threshold_start=-25,
    threshold_end=-35,
    buffer_start_ms=100,
    buffer_end_ms=150
  )

  __remove_silence(
    data_src_dir='/datasets/THCHS-30-test-22050',
    data_dest_dir='/datasets/THCHS-30-test_nosil',
    chunk_size=5,
    threshold_start=-25,
    threshold_end=-35,
    buffer_start_ms=100,
    buffer_end_ms=150
  )
