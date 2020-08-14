import os

from tqdm import tqdm

from src.core.common import normalize_file
from src.pre.wav_pre_io import parse_data, save_data, already_exists, WavData, WavDataList, get_wavs_dir

def __normalize(base_dir: str, source_name: str, destination_name: str):
  if not already_exists(base_dir, destination_name):
    data = parse_data(base_dir, source_name)
    dest_dir = get_wavs_dir(base_dir, destination_name)
    result: WavDataList = []

    print("Normalizing...")
    values: WavData
    for values in tqdm(data):
      dest_wav_path = os.path.join(dest_dir, "{}_{}.wav".format(values.i, values.basename))
      
      normalize_file(
        in_path = values.wav,
        out_path = dest_wav_path
      )

      values.wav = dest_wav_path
      result.append(values)

    save_data(base_dir, destination_name, result)

def init_normalize_parser(parser):
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--source_name', type=str, help='THCHS dataset directory', required=True)
  parser.add_argument('--destination_name', type=str, help='THCHS destination directory', required=True)
  return __normalize

if __name__ == "__main__":
  __normalize(
    base_dir="/datasets/models/taco2pt_v2",
    source_name='thchs_16000kHz',
    destination_name='thchs_16000kHz_normalized'
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
