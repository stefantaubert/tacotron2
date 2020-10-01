# For copyright see LICENCE

import logging
from typing import Dict, Optional

import numpy as np
import torch

from src.core.common.audio import is_overamp, normalize_wav
from src.core.common.train import overwrite_custom_hparams
from src.core.waveglow.denoiser import Denoiser
from src.core.waveglow.train import CheckpointWaveglow, load_model


class Synthesizer():
  def __init__(self, checkpoint: CheckpointWaveglow, custom_hparams: Optional[Dict[str, str]], logger: logging.Logger):
    super().__init__()
    self._logger = logger

    hparams = checkpoint.get_hparams()
    hparams = overwrite_custom_hparams(hparams, custom_hparams)

    model = load_model(hparams, checkpoint.state_dict)
    model = model.remove_weightnorm(model)
    model = model.eval()

    denoiser = Denoiser(model)
    denoiser = denoiser.cuda()

    self.hparams = hparams
    self.model = model
    self.denoiser = denoiser

  def infer(self, mel, sigma: float, denoiser_strength: float) -> np.ndarray:
    with torch.no_grad():
      audio = self.model.infer(mel, sigma=sigma)
      if denoiser_strength > 0:
        assert self.denoiser
        audio = self.denoiser(audio, strength=denoiser_strength)
    audio = audio.squeeze()
    audio = audio.cpu()
    audio_np: np.ndarray = audio.numpy()

    if is_overamp(audio_np):
      self._logger.warn("Waveglow output was overamplified.")

    audio_np_normalized = normalize_wav(audio_np)

    return audio_np_normalized
