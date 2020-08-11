import os
from src.core.pre.ds import DsDataList
from src.core.pre.wav import WavDataList
from src.core.pre.mel import MelDataList
from src.cli.pre.paths import get_mel_subdir, get_mel_csv
from src.cli.pre.wav import load_wav_csv
from argparse import ArgumentParser

#region IO

def load_mel_csv(base_dir: str, ds_name: str, sub_name: str) -> MelDataList:
  data_path = get_mel_csv(base_dir, ds_name, sub_name)
  return MelDataList.load(data_path)
  
def save_mel_csv(base_dir: str, ds_name: str, sub_name: str, mel_data: MelDataList):
  data_path = get_mel_csv(base_dir, ds_name, sub_name)
  mel_data.save(data_path)

def mel_subdir_exists(base_dir: str, ds_name: str, sub_name: str):
  data_dir = get_mel_subdir(base_dir, ds_name, sub_name, create=False)
  return os.path.exists(data_dir)

#endregion

#region Processing wavs

def init_pre_parser(parser: ArgumentParser):
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--ds_name', type=str, required=True)
  parser.add_argument('--sub_name', type=str, required=True)
  parser.add_argument('--custom_hparams', type=str)
  return pre

def pre(base_dir: str, ds_name: str, sub_name: str, custom_hparams: str):
  if mel_subdir_exists(base_dir, ds_name, sub_name):
    print("Already exists.")
  else:
    from src.core.pre.mel.pre import process
    data = load_wav_csv(base_dir, ds_name, sub_name)
    #wav_data_dir = get_pre_ds_wav_subname_dir(base_dir, ds_name, sub_name, create=False)
    data_dir = get_mel_subdir(base_dir, ds_name, sub_name, create=True)
    mel_data = process(data, data_dir, custom_hparams)
    save_mel_csv(base_dir, ds_name, sub_name, mel_data)

#endregion

if __name__ == "__main__":
  pre(
    base_dir="/datasets/models/taco2pt_v2",
    ds_name="thchs",
    sub_name="22050kHz_normalized_nosil",
    custom_hparams="",
  )
