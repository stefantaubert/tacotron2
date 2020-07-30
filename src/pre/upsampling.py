import os

from tqdm import tqdm

from src.common.audio.utils import upsample
from src.paths import get_wavs_dir
from src.pre.wav_pre import parse_data, save_data


def __upsample_wavs(base_dir, source_name, destination_name, new_rate):
  data = parse_data(base_dir, source_name)
  dest_dir = get_wavs_dir(base_dir, destination_name)
  result = []

  print("Upsampling...")
  for i, values in tqdm(enumerate(data), total=len(data)):
    dest_wav_path = os.path.join(dest_dir, "{}.wav".format(i))
    wav_path = values[3] 
    upsample(wav_path, dest_wav_path, new_rate)
    values[3] = dest_wav_path
    result.append(values)
  save_data(base_dir, destination_name, result)

def init_upsample_parser(parser):
  parser.add_argument('--data_src_dir', type=str, help='THCHS dataset directory', required=True)
  parser.add_argument('--data_dest_dir', type=str, help='THCHS destination directory', required=True)
  parser.add_argument('--new_rate', type=int, default=22050)
  return __upsample_wavs

if __name__ == "__main__":
  __upsample_wavs(
    data_src_dir='/datasets/thchs_wav',
    data_dest_dir='/datasets/thchs_16bit_22050kHz',
    new_rate=22050
  )

  __upsample_wavs(
    data_src_dir='/datasets/THCHS-30-test',
    data_dest_dir='/datasets/THCHS-30-test-22050',
    new_rate=22050
  )
