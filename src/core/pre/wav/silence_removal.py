import os
from tqdm import tqdm
from src.common.audio.utils import remove_silence_file
from src.core.pre.wav.data import WavData, WavDataList

def process(data: WavDataList, dest_dir: str, chunk_size: int, threshold_start: float, threshold_end: float, buffer_start_ms: float, buffer_end_ms: float) -> WavDataList:
  assert os.path.isdir(dest_dir)
  result: WavDataList = []

  values: WavData
  for values in tqdm(data):
    dest_wav_path = os.path.join(dest_dir, f"{values!r}.wav")
    new_duration = remove_silence_file(
      in_path = values.wav,
      out_path = dest_wav_path,
      chunk_size = chunk_size,
      threshold_start = threshold_start,
      threshold_end = threshold_end,
      buffer_start_ms = buffer_start_ms,
      buffer_end_ms = buffer_end_ms
    )
    result.append(WavData(values.entry_id, dest_wav_path, new_duration, values.sr))

  return result
