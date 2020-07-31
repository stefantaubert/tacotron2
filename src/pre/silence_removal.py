import os

from tqdm import tqdm

from src.paths import get_wavs_dir
from src.common.audio.remove_silence import remove_silence
from src.common.audio.utils import get_duration
from src.pre.wav_data import parse_data, save_data, get_path, set_path, set_duration, get_basename, get_id, already_exists

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
    result = []

    print("Removing silence...")
    for values in tqdm(data):
      dest_wav_path = os.path.join(dest_dir, "{}_{}.wav".format(get_id(values), get_basename(values)))
      wav_path = get_path(values)
      
      remove_silence(
        in_path = wav_path,
        out_path = dest_wav_path,
        chunk_size = chunk_size,
        threshold_start = threshold_start,
        threshold_end = threshold_end,
        buffer_start_ms = buffer_start_ms,
        buffer_end_ms = buffer_end_ms
      )

      set_path(values, dest_wav_path)
      new_duration = get_duration(dest_wav_path)
      set_duration(values, new_duration)
      result.append(values)

    save_data(base_dir, destination_name, result)

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
  __remove_silence(
    base_dir="/datasets/models/taco2pt_v2",
    source_name='thchs_22050kHz',
    destination_name='thchs_22050kHz_nosil',
    chunk_size=5,
    threshold_start=-25,
    threshold_end=-35,
    buffer_start_ms=100,
    buffer_end_ms=150
  )

  __remove_silence(
    base_dir="/datasets/models/taco2pt_v2",
    source_name='thchs_kaldi_22050kHz',
    destination_name='thchs_kaldi_22050kHz_nosil',
    chunk_size=5,
    threshold_start=-25,
    threshold_end=-35,
    buffer_start_ms=100,
    buffer_end_ms=150
  )
