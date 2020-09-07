from src.core.common.utils import cosine_dist_mels
from src.core.common.audio import normalize_wav
from src.core.common.taco_stft import TacotronSTFT
from src.core.pre.merge_ds import PreparedData
from typing import Optional
from src.core.waveglow.synthesizer import Synthesizer
import torch
import logging
import numpy as np

def get_logger():
  return logging.getLogger("infer")

_logger = get_logger()

def validate(entry: PreparedData, custom_hparams: Optional[str], denoiser_strength: float, sigma: float, checkpoint_path: str):
  _logger.info(f"Validating {entry.wav_path}...")
  return infer_core(entry.wav_path, custom_hparams, denoiser_strength, sigma, checkpoint_path)

def infer(wav_path: str, custom_hparams: Optional[str], denoiser_strength: float, sigma: float, checkpoint_path: str):
  _logger.info(f"Inferring {wav_path}...")
  return infer_core(wav_path, custom_hparams, denoiser_strength, sigma, checkpoint_path)

def infer_core(wav_path: str, custom_hparams: Optional[str], denoiser_strength: float, sigma: float, checkpoint_path: str):

  synth = Synthesizer(checkpoint_path, custom_hparams, _logger)

  # if is_fp16:
  #   from apex import amp
  #   waveglow, _ = amp.initialize(waveglow, [], opt_level="O3")

  taco_stft = TacotronSTFT.fromhparams(synth.hparams)

  mel = taco_stft.get_mel_tensor_from_file(wav_path)
  mel_var = torch.autograd.Variable(mel)
  mel_var = mel_var.cuda()
  mel_var = mel_var.unsqueeze(0)

  #mel = mel.half() if is_fp16 else mel

  audio = synth.infer(mel_var, sigma, denoiser_strength)
  audio = normalize_wav(audio)
  audio_tensor = torch.FloatTensor(audio)
  mel_pred = taco_stft.get_mel_tensor(audio_tensor)

  orig_np = mel.cpu().numpy()
  pred_np = mel_pred.numpy()

  score = cosine_dist_mels(orig_np, pred_np)
  _logger.info(f"Cosine similarity is: {score*100}%")

  #score, diff_img = compare_mels(a, b)
  return audio, mel_pred, mel
