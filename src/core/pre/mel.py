"""
input: wav data
output: mel data
"""
from dataclasses import dataclass
from logging import getLogger
from typing import Any, Dict, Optional

from src.core.common.taco_stft import TSTFTHParams, TacotronSTFT
from src.core.common.train import overwrite_custom_hparams
from src.core.common.utils import GenericList
from src.core.pre.wav import WavDataList

@dataclass()
class MelData:
  entry_id: int
  mel_path: str
  n_mel_channels: int


class MelDataList(GenericList[MelData]):
  pass


def process(data: WavDataList, custom_hparams: Optional[Dict[str, str]], save_callback: Any) -> MelDataList:
  result = MelDataList()
  hparams = TSTFTHParams()
  hparams = overwrite_custom_hparams(hparams, custom_hparams)
  mel_parser = TacotronSTFT(hparams, logger=getLogger())

  for wav_entry in data.items(True):
    mel_tensor = mel_parser.get_mel_tensor_from_file(wav_entry.wav)
    path = save_callback(wav_entry=wav_entry, mel_tensor=mel_tensor)
    result.append(MelData(wav_entry.entry_id, path, hparams.n_mel_channels))

  return result
