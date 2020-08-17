from src.core.waveglow.synthesizer import Synthesizer
import torch
from src.core.common import TacotronSTFT
import logging
from src.core.pre import PreparedData

def get_logger():
  return logging.getLogger("infer")

_logger = get_logger()

def infer(wav_path: str, custom_hparams: str, denoiser_strength: float, sigma: float, checkpoint_path: str):
  _logger.info(f"Inferring {wav_path}...")
  return infer_core(wav_path, custom_hparams, denoiser_strength, sigma, checkpoint_path)

def validate(entry: PreparedData, custom_hparams: str, denoiser_strength: float, sigma: float, checkpoint_path: str):
  _logger.info(f"Validating {entry.wav_path}...")
  return infer_core(entry.wav_path, custom_hparams, denoiser_strength, sigma, checkpoint_path)

def infer_core(wav_path: str, custom_hparams: str, denoiser_strength: float, sigma: float, checkpoint_path: str):

  synth = Synthesizer(checkpoint_path, custom_hparams, _logger)
  
  # if is_fp16:
  #   from apex import amp
  #   waveglow, _ = amp.initialize(waveglow, [], opt_level="O3")

  taco_stft = TacotronSTFT(synth.hparams)

  mel = taco_stft.get_mel_tensor_from_file(wav_path)
  mel = mel.cuda()
  mel = torch.autograd.Variable(mel)
  mel = mel.unsqueeze(0)

  #mel = mel.half() if is_fp16 else mel

  audio = synth.infer_mel(mel, sigma, denoiser_strength)

  return audio
