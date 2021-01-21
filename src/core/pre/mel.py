"""
input: wav data
output: mel data
"""
import os
from dataclasses import dataclass
from logging import getLogger
from typing import Any, Callable, Dict, Optional

from src.core.common.taco_stft import TacotronSTFT, TSTFTHParams
from src.core.common.train import overwrite_custom_hparams
from src.core.common.utils import GenericList
from src.core.pre.wav import WavData, WavDataList
from torch import Tensor


@dataclass()
class MelData:
  entry_id: int
  relative_mel_path: str
  n_mel_channels: int


class MelDataList(GenericList[MelData]):
  pass


def process(data: WavDataList, wav_dir: str, custom_hparams: Optional[Dict[str, str]], save_callback: Callable[[WavData, Tensor], str]) -> MelDataList:
  result = MelDataList()
  hparams = TSTFTHParams()
  hparams = overwrite_custom_hparams(hparams, custom_hparams)
  mel_parser = TacotronSTFT(hparams, logger=getLogger())

  for wav_entry in data.items(True):
    absolute_wav_path = os.path.join(wav_dir, wav_entry.relative_wav_path)
    mel_tensor = mel_parser.get_mel_tensor_from_file(absolute_wav_path)
    path = save_callback(wav_entry=wav_entry, mel_tensor=mel_tensor)
    mel_data = MelData(wav_entry.entry_id, path, hparams.n_mel_channels)
    result.append(mel_data)

  return result
