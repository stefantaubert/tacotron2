from src.core.waveglow.synthesizer import Synthesizer
import torch
from src.core.common import TacotronSTFT, compare_mels, normalize_wav
import logging
from src.core.pre import PreparedData
from scipy.spatial.distance import cosine

def get_logger():
  return logging.getLogger("infer")

_logger = get_logger()

def validate(entry: PreparedData, custom_hparams: str, denoiser_strength: float, sigma: float, checkpoint_path: str):
  _logger.info(f"Validating {entry.wav_path}...")
  return infer_core(entry.wav_path, custom_hparams, denoiser_strength, sigma, checkpoint_path)

def infer(wav_path: str, custom_hparams: str, denoiser_strength: float, sigma: float, checkpoint_path: str):
  _logger.info(f"Inferring {wav_path}...")
  return infer_core(wav_path, custom_hparams, denoiser_strength, sigma, checkpoint_path)

def infer_core(wav_path: str, custom_hparams: str, denoiser_strength: float, sigma: float, checkpoint_path: str):

  synth = Synthesizer(checkpoint_path, custom_hparams, _logger)
  
  # if is_fp16:
  #   from apex import amp
  #   waveglow, _ = amp.initialize(waveglow, [], opt_level="O3")

  taco_stft = TacotronSTFT.fromhparams(synth.hparams)

  mel = taco_stft.get_mel_tensor_from_file(wav_path)
  mel = mel.cuda()
  mel_var = torch.autograd.Variable(mel)
  mel_var = mel_var.unsqueeze(0)

  #mel = mel.half() if is_fp16 else mel

  audio = synth.infer_mel(mel_var, sigma, denoiser_strength)
  audio = normalize_wav(audio)
  audio_tensor = torch.FloatTensor(audio)
  mel_pred = taco_stft.get_mel_tensor(audio_tensor)

  pred_np = mel.cpu().numpy()
  orig_np = mel_pred.numpy()
  score = cosine(pred_np, orig_np)
  _logger.info(f"Cosine similarity is: {score*100}%")

  #score, diff_img = compare_mels(a, b)
  return audio, mel_pred, mel
