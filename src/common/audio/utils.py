import random
from math import inf, log10

import numpy as np
from resampy import resample as resamply_resample
from scipy.io.wavfile import read, write

import torch


float32_64_min_wav = -1.0
float32_64_max_wav = 1.0
int16_min_wav = np.iinfo(np.int16).min # -32768 = -(2**15)
int16_max_wav = np.iinfo(np.int16).max # 32767 = 2**15 - 1
int32_min_wav = np.iinfo(np.int32).min # -2147483648 = -(2**31)
int32_max_wav = np.iinfo(np.int32).max # 2147483647 = 2**31 - 1

def get_dBFS(wav, max_value) -> float:
  new = np.sqrt(np.mean((wav / max_value)**2))
  #new = np.mean(wav) / max_value
  if new == 0:
    return -inf
  else:
    result = 20 * log10(new)
    return result

def detect_leading_silence(wav: np.array, silence_threshold: float, chunk_size: int, buffer: int):
    assert chunk_size > 0
    if chunk_size > len(wav):
      chunk_size = len(wav)

    trim = 0
    max_value = -get_min_value(wav.dtype)
    while get_dBFS(wav[trim:trim + chunk_size], max_value) < silence_threshold and trim < len(wav):
      trim += chunk_size

    if trim >= buffer:
      trim -= buffer
    else:
      trim = 0

    return trim

def remove_silence(
    wav,
    chunk_size: int,
    threshold_start: float,
    threshold_end: float,
    buffer_start: float,
    buffer_end: float
  ):

  start_trim = detect_leading_silence(
    wav=wav,
    silence_threshold=threshold_start,
    chunk_size=chunk_size,
    buffer=buffer_start
  )

  wav_reversed = wav[::-1]
  end_trim = detect_leading_silence(
    wav=wav_reversed,
    silence_threshold=threshold_end,
    chunk_size=chunk_size,
    buffer=buffer_end
  )
  
  wav = wav[start_trim:len(wav) - end_trim]
  return wav

def ms_to_samples(ms, sampling_rate):
  res = int(ms * sampling_rate / 1000)
  return res


def remove_silence_file(
    in_path: str,
    out_path: str,
    chunk_size: int,
    threshold_start: float,
    threshold_end: float,
    buffer_start_ms: float,
    buffer_end_ms: float
  ):

  sampling_rate, wav = read(in_path)

  buffer_start = ms_to_samples(buffer_start_ms, sampling_rate)
  buffer_end = ms_to_samples(buffer_end_ms, sampling_rate)

  wav = remove_silence(
    wav = wav,
    chunk_size = chunk_size,
    threshold_start = threshold_start,
    threshold_end = threshold_end,
    buffer_start = buffer_start,
    buffer_end = buffer_end
  )
  new_duration = get_duration_s(wav, sampling_rate)

  write(filename=out_path, rate=sampling_rate, data=wav)

  return new_duration

def float_to_wav(wav, path, dtype=np.int16, normalize=True, sample_rate=22050):
  # denoiser_out is float64
  # waveglow_out is float32

  wav = convert_wav(wav, dtype)

  if normalize:
    wav = normalize_wav(wav)

  write(filename=path, rate=sample_rate, data=wav)

def convert_wav(wav, to_dtype):
  '''
  if the wav is overamplified the result will also be overamplified.
  '''
  if wav.dtype != to_dtype:
    wav = wav / -get_min_value(wav.dtype) * get_max_value(to_dtype)
    if to_dtype == np.int16 or to_dtype == np.int32:
      # the default seems to be np.fix instead of np.round on wav.astype()
      wav = np.round(wav, 0)
    wav = wav.astype(to_dtype)

  return wav

def fix_overamplification(wav):
  is_overamplified = is_overamp(wav)
  if is_overamplified:
    wav = normalize_wav(wav)
  return wav

def get_max_value(dtype):
  # see wavfile.write() max positive eg. on 16-bit PCM is 32767
  if dtype==np.int16: return int16_max_wav
  elif dtype==np.int32: return int32_max_wav
  elif dtype==np.float32 or dtype == np.float64: return float32_64_max_wav
  else: assert False

def get_min_value(dtype):
  if dtype==np.int16: return int16_min_wav
  elif dtype==np.int32: return int32_min_wav
  elif dtype==np.float32 or dtype == np.float64: return float32_64_min_wav
  else: assert False

def normalize_file(in_path, out_path):
  sampling_rate, wav = read(in_path)
  wav = normalize_wav(wav)
  write(filename=out_path, rate=sampling_rate, data=wav)

def normalize_wav(wav):
  if wav.dtype == np.int16 and np.min(wav) == get_min_value(np.int16):
    return wav
  elif wav.dtype == np.int32 and np.min(wav) == get_min_value(np.int32):
    return wav
  
  wav_abs = np.abs(wav)
  max_val = np.max(wav_abs)
  is_div_by_zero = max_val == 0
  max_possible_value = get_max_value(wav.dtype)
  is_already_normalized = max_val == max_possible_value
  # on int16 resulting min wav value would be max. -32767 not -32768 (which would be possible with wavfile.write) maybe later TODO

  if not is_already_normalized and not is_div_by_zero:
    orig_dtype = wav.dtype
    wav_float = wav.astype(np.float32)
    wav_float = wav_float * max_possible_value / max_val
    if orig_dtype == np.int16 or orig_dtype == np.int32:
      # the default seems to be np.fix instead of np.round on wav.astype()
      # 32766.998 gets 32767 because of float unaccuracy
      wav_float = np.round(wav_float, 0)
    wav = wav_float.astype(orig_dtype)
  
  assert np.max(np.abs(wav)) == max_possible_value or np.max(np.abs(wav)) == 0

  return wav

def wav_to_float32(path: str) -> (np.float32, int):
  sampling_rate, wav = read(path)
  wav = convert_wav(wav, np.float32)
  return wav, sampling_rate

def is_overamp(wav):
  lowest_value = get_min_value(wav.dtype)
  highest_value = get_max_value(wav.dtype)
  wav_min = np.min(wav)
  wav_max = np.max(wav)
  is_overamplified = wav_min < lowest_value or wav_max > highest_value
  return is_overamplified

def resample_core(wav, sr, new_rate):
  if sr != new_rate:
    origin_dtype = wav.dtype
    wav_float = convert_wav(wav, np.float32)
    wav_float = resamply_resample(wav_float, sr, new_rate)
    # if a.min was -1 before resample it would be smaller than -1 (bug in resample)
    wav_float = fix_overamplification(wav_float)
    wav = convert_wav(wav_float, origin_dtype)
  return wav

def upsample(origin, dest, new_rate):
  sampling_rate, wav = read(origin)
  wav = resample_core(wav, sampling_rate, new_rate)
  write(filename=dest, rate=new_rate, data=wav)

def get_duration_s(wav, sampling_rate) -> float:
  duration = len(wav) / sampling_rate
  return duration

def get_duration_s_file(wav_path) -> float:
  sampling_rate, wav = read(wav_path)
  return get_duration_s(wav, sampling_rate)

def mel_to_numpy(mel):
  mel = mel.squeeze(0)
  mel = mel.cpu()
  mel = mel.numpy()
  return mel

def wav_to_float32_tensor(path: str) -> (torch.float32, int):
  wav, sampling_rate = wav_to_float32(path)
  wav_tensor = torch.FloatTensor(wav)

  return wav_tensor, sampling_rate

def get_wav_tensor_segment(wav_tensor: torch.float32, segment_length: int) -> torch.float32:
  if wav_tensor.size(0) >= segment_length:
    max_audio_start = wav_tensor.size(0) - segment_length
    audio_start = random.randint(0, max_audio_start)
    wav_tensor = wav_tensor[audio_start:audio_start+segment_length]
  else:
    fill_size = segment_length - wav_tensor.size(0)
    wav_tensor = torch.nn.functional.pad(wav_tensor, (0, fill_size), 'constant').data
  
  return wav_tensor

if __name__ == "__main__":
  normalize_file("/datasets/thchs_wav/wav/train/A2/A2_21.wav", "/tmp/A2_21.wav")
  import tempfile
  dest = tempfile.mktemp("-A13_224.wav")
  new_dur = remove_silence_file(
    in_path = "/datasets/thchs_wav/wav/train/A13/A13_224.wav",
    out_path = dest,
    threshold_start = -20,
    threshold_end = -25,
    chunk_size = 5,
    buffer_start_ms = 25,
    buffer_end_ms = 25
  )

  wav, sr = wav_to_float32("/datasets/thchs_wav/wav/train/A22/A22_107.wav")
  trim = detect_leading_silence(wav, silence_threshold=-25, chunk_size=5, buffer_ms=100)
  wav = wav[trim:]
  float_to_wav(wav, "/tmp/A22_107_nosil.wav", sample_rate=sr)
  #upsample("/datasets/Report/08/03/B4_322.wav", "/tmp/B4_322.wav", 22050)
  upsample("/datasets/thchs_wav/wav/test/D31/D31_881.wav", "/tmp/D31_881.wav", 22050)
  #ewav, _ = wav_to_float32("/datasets/models/taco2pt_v2/thchs_ipa_warm_mapped_all_tones/inference/validation_2020-07-27_08-54-17_D31_769_D31_50777/validation_2020-07-27_08-54-17_D31_769_D31_50777_inferred.wav")
  #float_to_wav(wav, "/tmp/out.wav", dtype=np.int16, normalize=True, sample_rate=22050)
