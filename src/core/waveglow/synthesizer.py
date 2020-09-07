# For copyright see LICENCE

import os
from typing import Optional

import torch
from src.core.waveglow.denoiser import Denoiser
from src.core.waveglow.hparams import create_hparams
from src.core.waveglow.train import load_model
from src.core.common import is_overamp
import logging
import numpy as np

class Synthesizer():
  def __init__(self, checkpoint_path: str, custom_hparams: Optional[str], logger: logging.Logger):
    super().__init__()
    assert os.path.isfile(checkpoint_path)
    self._logger = logger
    self._logger.info(f"Loading waveglow model from: {checkpoint_path}")

    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model_state_dict = checkpoint_dict['state_dict']
    # TODO pass waveglow hparams in tacotron with arguments (only required if used non default hparams)
    hparams = create_hparams(custom_hparams)
    model = load_model(hparams)
    model.load_state_dict(model_state_dict)

    model = model.remove_weightnorm(model)
    model = model.cuda()
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
        audio = self.denoiser.forward(audio, strength=denoiser_strength)
    audio = audio.squeeze()
    audio = audio.cpu()
    audio_np: np.ndarray = audio.numpy()

    if is_overamp(audio_np):
      self._logger.warn("Waveglow output is overamplified.")

    return audio_np
