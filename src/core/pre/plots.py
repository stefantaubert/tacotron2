"""
input: wav data
output: mel data
"""
import os
from logging import getLogger
from typing import Any, Callable, Dict, List, Optional

from src.core.common.taco_stft import TacotronSTFT, TSTFTHParams
from src.core.common.train import overwrite_custom_hparams
from src.core.pre.ds import DsData, DsDataList
from src.core.pre.wav import WavData, WavDataList


def process(data: WavDataList, ds: DsDataList, wav_dir: str, custom_hparams: Optional[Dict[str, str]], save_callback: Callable[[WavData, DsData], str]) -> None:
  hparams = TSTFTHParams()
  hparams = overwrite_custom_hparams(hparams, custom_hparams)
  mel_parser = TacotronSTFT(hparams, logger=getLogger())

  all_paths: List[str] = []
  for wav_entry, ds_entry in zip(data.items(True), ds.items(True)):
    absolute_wav_path = os.path.join(wav_dir, wav_entry.relative_wav_path)
    mel_tensor = mel_parser.get_mel_tensor_from_file(absolute_wav_path)
    absolute_path = save_callback(wav_entry=wav_entry, ds_entry=ds_entry, mel_tensor=mel_tensor)
    all_paths.append(absolute_path)
  return all_paths
