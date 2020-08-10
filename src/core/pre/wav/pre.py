"""
calculate wav duration
"""

from tqdm import tqdm
from src.common.audio.utils import get_duration_s_file, get_duration_s
from scipy.io.wavfile import read
from src.core.pre.wav.data import WavData, WavDataList
from src.core.pre.ds.data import DsDataList, DsData

def process(data: DsDataList) -> WavDataList:
  result: WavDataList = []
  
  values: DsData
  for values in tqdm(data):
    sampling_rate, wav = read(values.wav_path)
    duration = get_duration_s(wav, sampling_rate)
    result.append(WavData(values.entry_id, values.wav_path, duration, sampling_rate))

  return result
