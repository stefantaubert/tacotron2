from logging import Logger
from typing import Optional

import torch

from src.core.common.taco_stft import TacotronSTFT
from src.core.common.utils import cosine_dist_mels
from src.core.waveglow.synthesizer import Synthesizer
from src.core.waveglow.train import CheckpointWaveglow


def infer(wav_path: str, checkpoint: CheckpointWaveglow, custom_hparams: Optional[str], denoiser_strength: float, sigma: float, logger: Logger):
  synth = Synthesizer(
    checkpoint=checkpoint,
    custom_hparams=custom_hparams,
    logger=logger
  )

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
  audio_tensor = torch.FloatTensor(audio)
  mel_pred = taco_stft.get_mel_tensor(audio_tensor)

  orig_np = mel.cpu().numpy()
  pred_np = mel_pred.numpy()

  score = cosine_dist_mels(orig_np, pred_np)
  logger.info(f"Cosine similarity is: {score*100}%")

  #score, diff_img = compare_mels(a, b)
  return audio, synth.hparams.sampling_rate, pred_np, orig_np
