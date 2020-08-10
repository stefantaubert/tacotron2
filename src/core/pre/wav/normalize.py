import os
from tqdm import tqdm
from src.common.audio.utils import normalize_file
from src.core.pre.wav.data import WavData, WavDataList

def process(data: WavDataList, dest_dir: str) -> WavDataList:
  assert os.path.isdir(dest_dir)
  result: WavDataList = []
  
  values: WavData
  for values in tqdm(data):
    dest_wav_path = os.path.join(dest_dir, f"{values!r}.wav")
    normalize_file(values.wav, dest_wav_path)
    result.append(WavData(values.entry_id, dest_wav_path, values.duration, values.sr))

  return result
