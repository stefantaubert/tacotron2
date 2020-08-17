# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#    * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#    * Neither the name of the NVIDIA CORPORATION nor the
#    names of its contributors may be used to endorse or promote products
#    derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************
import os

import torch
from src.core.waveglow.denoiser import Denoiser
from src.core.waveglow.hparams import create_hparams
from src.core.waveglow.train import load_model
from src.core.common import is_overamp
import logging
import numpy as np

class Synthesizer():
  def __init__(self, checkpoint_path: str, custom_hparams: str = None, logger: logging.Logger = logging.getLogger()):
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

  def infer_mel(self, mel, sigma: float, denoiser_strength: float) -> np.ndarray:
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
