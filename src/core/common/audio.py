import random
from typing import Tuple

import numpy as np
import torch
from audio_utils import wav_to_float32


def mel_to_numpy(mel: torch.Tensor) -> np.ndarray:
  mel = mel.squeeze(0)
  mel = mel.cpu()
  mel_np: np.ndarray = mel.numpy()
  return mel_np


def wav_to_float32_tensor(path: str) -> Tuple[torch.Tensor, int]:
  wav, sampling_rate = wav_to_float32(path)
  wav_tensor = torch.FloatTensor(wav)

  return wav_tensor, sampling_rate


def get_wav_tensor_segment(wav_tensor: torch.Tensor, segment_length: int) -> torch.Tensor:
  if wav_tensor.size(0) >= segment_length:
    max_audio_start = wav_tensor.size(0) - segment_length
    audio_start = random.randint(0, max_audio_start)
    wav_tensor = wav_tensor[audio_start:audio_start + segment_length]
  else:
    fill_size = segment_length - wav_tensor.size(0)
    wav_tensor = torch.nn.functional.pad(wav_tensor, (0, fill_size), 'constant').data

  return wav_tensor


if __name__ == "__main__":
  from scipy.io.wavfile import read

  #sr, wav = read("/datasets/NNLV_pilot/40mins/audio/0.wav")
  #sr, wav = read("/data/datasets/l2arctic/suitcase_corpus/wav/tlv.wav")
  x = wav_to_float32("/data/datasets/l2arctic/suitcase_corpus/wav/tlv.wav")
  # wav = normalize_wav(wav)
  # wav = stereo_to_mono(wav)
  # wav = resample_core(wav, sr, 22050)
  # write("/tmp/test.wav", 22050, wav)
  # normalize_file("/datasets/thchs_wav/wav/train/A2/A2_21.wav", "/tmp/A2_21.wav")
  # import tempfile
  # dest = tempfile.mktemp("-A13_224.wav")
  # new_dur = remove_silence_file(
  #   in_path="/datasets/thchs_wav/wav/train/A13/A13_224.wav",
  #   out_path=dest,
  #   threshold_start=-20,
  #   threshold_end=-25,
  #   chunk_size=5,
  #   buffer_start_ms=25,
  #   buffer_end_ms=25
  # )

  #wav, sr = wav_to_float32("/datasets/thchs_wav/wav/train/A22/A22_107.wav")
  #trim = detect_leading_silence(wav, silence_threshold=-25, chunk_size=5, buffer=100)
  #wav = wav[trim:]
  #float_to_wav(wav, "/tmp/A22_107_nosil.wav", sample_rate=sr)
  #upsample("/datasets/Report/08/03/B4_322.wav", "/tmp/B4_322.wav", 22050)
  #upsample("/datasets/thchs_wav/wav/test/D31/D31_881.wav", "/tmp/D31_881.wav", 22050)
  #ewav, _ = wav_to_float32("/datasets/models/taco2pt_v2/thchs_ipa_warm_mapped_all_tones/inference/validation_2020-07-27_08-54-17_D31_769_D31_50777/validation_2020-07-27_08-54-17_D31_769_D31_50777_inferred.wav")
  #float_to_wav(wav, "/tmp/out.wav", dtype=np.int16, normalize=True, sample_rate=22050)
