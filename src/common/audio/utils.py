import librosa
import torch
import numpy as np
from scipy.io.wavfile import read, write
import random

float32_64_max_wav = 1.0
int16_max_wav = np.iinfo(np.int16).max # 32767 = 2**15 - 1
int32_max_wav = np.iinfo(np.int32).max # 2147483647 = 2**31 - 1

def float_to_wav(wav, path, dtype=np.int16, normalize=True, sample_rate=22050):
  # denoiser_out is float64
  # waveglow_out is float32
  if wav.dtype != np.float64 and wav.dtype != np.float32:
      raise Exception("Not supported.")
  # see wavfile.write() max positive eg. on 16-bit PCM is 32767
  if dtype==np.int16:
    max_possible_value = int16_max_wav
  elif dtype==np.int32:
    max_possible_value = int32_max_wav
  elif dtype==np.float32:
    max_possible_value = float32_64_max_wav

  wav = wav / float32_64_max_wav * max_possible_value

  if normalize:
    # resulting min wav value would be max. -32767 not -32768 (which would be possible with wavfile.write) maybe later TODO
    max_val = np.max(np.abs(wav))
    is_div_by_zero = max_val == 0
    if not is_div_by_zero:
      amp_factor = max_possible_value / max_val
      wav *= amp_factor

  wav = wav.astype(dtype)
  write(filename=path, rate=sample_rate, data=wav)

def wav_to_float32(path: str) -> (np.float32, int):
  sampling_rate, wav = read(path)

  if wav.dtype == np.int16:
    x = int16_max_wav
  else:
    raise Exception("Bitdepth not supported.")

  wav = wav / x * float32_64_max_wav
  wav = wav.astype(np.float32)
  return wav, sampling_rate

def upsample(origin, dest, new_rate):
  new_data, _ = librosa.load(origin, sr=new_rate, mono=True, dtype=np.float32)
  float_to_wav(wav=new_data, path=dest, sample_rate=new_rate, normalize=False)

def get_duration(wav_path) -> float:
  sampling_rate, wav = read(wav_path)
  duration = len(wav) / sampling_rate
  return duration

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
  wav, _ = wav_to_float32("/datasets/models/taco2pt_v2/thchs_ipa_warm_mapped_all_tones/inference/validation_2020-07-27_08-54-17_D31_769_D31_50777/validation_2020-07-27_08-54-17_D31_769_D31_50777_inferred.wav")
  float_to_wav(wav, "/tmp/out.wav", dtype=np.int16, normalize=True, sample_rate=22050)