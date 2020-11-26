"""
input: wav data
output: mel data
"""
from logging import getLogger
from typing import Any, Dict, List, Optional

from src.core.common.taco_stft import TacotronSTFT, TSTFTHParams
from src.core.common.train import overwrite_custom_hparams
from src.core.pre.ds import DsDataList
from src.core.pre.wav import WavDataList


def process(data: WavDataList, ds: DsDataList, custom_hparams: Optional[Dict[str, str]], save_callback: Any) -> None:
  hparams = TSTFTHParams()
  hparams = overwrite_custom_hparams(hparams, custom_hparams)
  mel_parser = TacotronSTFT(hparams, logger=getLogger())

  all_paths: List[str] = []
  for wav_entry, ds_entry in zip(data.items(True), ds.items(True)):
    mel_tensor = mel_parser.get_mel_tensor_from_file(wav_entry.wav)
    path = save_callback(wav_entry=wav_entry, ds_entry=ds_entry, mel_tensor=mel_tensor)
    all_paths.append(path)
  return all_paths
