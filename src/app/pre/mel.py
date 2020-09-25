import os
from functools import partial
from typing import Dict, Optional

import torch
from torch import Tensor

from src.app.pre.ds import get_ds_dir
from src.app.pre.wav import get_wav_dir, load_wav_csv
from src.core.common.train import get_pytorch_filename
from src.core.common.utils import get_chunk_name, get_subdir
from src.core.pre.mel import MelData, MelDataList, process
from src.core.pre.wav import WavData

MEL_DATA_CSV = "data.csv"
CHUNK_SIZE = 500


def _get_mel_root_dir(ds_dir: str, create: bool = False):
  return get_subdir(ds_dir, "mel", create)


def get_mel_dir(ds_dir: str, mel_name: str, create: bool = False):
  return get_subdir(_get_mel_root_dir(ds_dir, create), mel_name, create)


def load_mel_csv(mel_dir: str) -> MelDataList:
  path = os.path.join(mel_dir, MEL_DATA_CSV)
  return MelDataList.load(MelData, path)


def save_mel_csv(mel_dir: str, mel_data: MelDataList):
  assert os.path.isdir(mel_dir)
  path = os.path.join(mel_dir, MEL_DATA_CSV)
  mel_data.save(path)


def save_mel(dest_dir: str, data_len: int, wav_entry: WavData, mel_tensor: Tensor) -> str:
  chunk_dir = os.path.join(dest_dir, get_chunk_name(
    wav_entry.entry_id, chunksize=CHUNK_SIZE, maximum=data_len - 1))
  os.makedirs(chunk_dir, exist_ok=True)
  dest_mel_path = os.path.join(chunk_dir, get_pytorch_filename(repr(wav_entry)))
  torch.save(mel_tensor, dest_mel_path)
  return dest_mel_path


def preprocess_mels(base_dir: str, ds_name: str, wav_name: str, custom_hparams: Optional[Dict[str, str]] = None):
  print("Preprocessing mels...")
  ds_dir = get_ds_dir(base_dir, ds_name)
  mel_dir = get_mel_dir(ds_dir, wav_name)
  if os.path.isdir(mel_dir):
    print("Already exists.")
  else:
    wav_dir = get_wav_dir(ds_dir, wav_name)
    assert os.path.isdir(wav_dir)
    data = load_wav_csv(wav_dir)
    assert len(data) > 0
    save_callback = partial(save_mel, dest_dir=mel_dir, data_len=len(data))
    mel_data = process(data, custom_hparams, save_callback)
    save_mel_csv(mel_dir, mel_data)


if __name__ == "__main__":

  preprocess_mels(
    base_dir="/datasets/models/taco2pt_v5",
    ds_name="thchs",
    wav_name="22050kHz_normalized_nosil"
  )

  preprocess_mels(
    base_dir="/datasets/models/taco2pt_v5",
    ds_name="ljs",
    wav_name="22050kHz"
  )
