import os

from tqdm import tqdm

from src.common.audio.utils import upsample
from src.pre.wav_pre_io import parse_data, save_data, already_exists, WavData, WavDataList, get_wavs_dir

def __upsample_wavs(base_dir: str, source_name: str, destination_name: str, new_rate: int):
  if not already_exists(base_dir, destination_name):
    data = parse_data(base_dir, source_name)
    dest_dir = get_wavs_dir(base_dir, destination_name)
    result: WavDataList = []

    print("Upsampling...")
    values: WavData
    for values in tqdm(data):
      dest_wav_path = os.path.join(dest_dir, "{}_{}.wav".format(values.i, values.basename))
      # todo assert not is_overamp
      upsample(values.wav, dest_wav_path, new_rate)
      values.wav = dest_wav_path
      result.append(values)
      
    save_data(base_dir, destination_name, result)

def init_upsample_parser(parser):
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--source_name', type=str, help='dataset directory', required=True)
  parser.add_argument('--destination_name', type=str, help='destination directory', required=True)
  parser.add_argument('--new_rate', type=int, default=22050)
  return __upsample_wavs

if __name__ == "__main__":
  __upsample_wavs(
    base_dir="/datasets/models/taco2pt_v2",
    source_name='thchs_16000kHz_normalized',
    destination_name='thchs_22050kHz_normalized',
    new_rate=22050,
  )

  # __upsample_wavs(
  #   base_dir="/datasets/models/taco2pt_v2",
  #   source_name='thchs_kaldi_16000kHz',
  #   destination_name='thchs_kaldi_22050kHz',
  #   new_rate=22050,
  # )
