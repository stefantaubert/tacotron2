import os
from src.core.pre import MelData

from src.app.pre.ds import get_ds_dir
from src.app.pre.wav import get_wav_dir, load_wav_csv
from src.core.common import get_subdir
from src.core.pre import (MelDataList, mels_preprocess)

_mel_data_csv = "data.csv"

def _get_mel_root_dir(ds_dir: str, create: bool = False):
  return get_subdir(ds_dir, "mel", create)

def get_mel_dir(ds_dir: str, mel_name: str, create: bool = False):
  return get_subdir(_get_mel_root_dir(ds_dir, create), mel_name, create)

def load_mel_csv(mel_dir: str) -> MelDataList:
  path = os.path.join(mel_dir, _mel_data_csv)
  return MelDataList.load(MelData, path)

def save_mel_csv(mel_dir: str, mel_data: MelDataList):
  path = os.path.join(mel_dir, _mel_data_csv)
  mel_data.save(path)

def preprocess_mels(base_dir: str, ds_name: str, wav_name: str, custom_hparams: str):
  print("Preprocessing mels...")
  ds_dir = get_ds_dir(base_dir, ds_name)
  mel_dir = get_mel_dir(ds_dir, wav_name)
  if os.path.isdir(mel_dir):
    print("Already exists.")
  else:
    wav_dir = get_wav_dir(ds_dir, wav_name)
    assert os.path.isdir(wav_dir)
    data = load_wav_csv(wav_dir)
    os.makedirs(mel_dir)
    mel_data = mels_preprocess(data, mel_dir, custom_hparams)
    save_mel_csv(mel_dir, mel_data)

if __name__ == "__main__":

  preprocess_mels(
    base_dir="/datasets/models/taco2pt_v5",
    ds_name="thchs",
    wav_name="22050kHz_normalized_nosil",
    custom_hparams="",
  )

  preprocess_mels(
    base_dir="/datasets/models/taco2pt_v5",
    ds_name="ljs",
    wav_name="22050kHz",
    custom_hparams="",
  )
