"""
input: wav data
output: mel data
"""
import os
import torch
from tqdm import tqdm
from src.common.audio.utils import normalize_file
from src.core.pre.wav.data import WavData, WavDataList
from src.core.pre.mel.data import MelData, MelDataList
from src.core.pre.mel.parser import MelParser, create_hparams

def process(data: WavDataList, dest_dir: str, custom_hparams: str) -> MelDataList:
  assert os.path.isdir(dest_dir)

  result: MelDataList = []
  hparms = create_hparams(custom_hparams)
  mel_parser = MelParser(hparms)

  values: WavData
  for values in tqdm(data):
    dest_mel_path = os.path.join(dest_dir, f"{values!r}.pt")
    mel_tensor = mel_parser.get_mel_tensor_from_file(values.wav)
    torch.save(mel_tensor, dest_mel_path)
    result.append(MelData(values.entry_id, dest_mel_path, values.duration, values.sr))

  return result
