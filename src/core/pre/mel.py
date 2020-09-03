"""
input: wav data
output: mel data
"""
import os
from dataclasses import dataclass
from src.core.common import GenericList

import torch
from tqdm import tqdm

from src.core.common import get_pytorch_filename
from src.core.common import get_chunk_name
from src.core.pre.wav import WavData, WavDataList
from src.core.common import TacotronSTFT, create_hparams


@dataclass()
class MelData:
  entry_id: int
  mel_path: str
  n_mel_channels: int


class MelDataList(GenericList[MelData]):
  pass


def process(data: WavDataList, dest_dir: str, custom_hparams: str) -> MelDataList:
  assert os.path.isdir(dest_dir)

  result = MelDataList()
  hp = create_hparams(custom_hparams)
  mel_parser = TacotronSTFT.fromhparams(hp)

  values: WavData
  for values in tqdm(data):
    chunk_dir = os.path.join(dest_dir, get_chunk_name(
      values.entry_id, chunksize=500, maximum=len(data) - 1))
    os.makedirs(chunk_dir, exist_ok=True)
    dest_mel_path = os.path.join(chunk_dir, get_pytorch_filename(repr(values)))
    mel_tensor = mel_parser.get_mel_tensor_from_file(values.wav)
    torch.save(mel_tensor, dest_mel_path)
    result.append(MelData(values.entry_id, dest_mel_path, hp.n_mel_channels))

  return result
