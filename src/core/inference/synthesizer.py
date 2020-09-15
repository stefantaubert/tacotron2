import logging
from typing import List, Optional, Tuple

import numpy as np
import torch

from src.core.tacotron.synthesizer import Synthesizer as TacoSynthesizer
from src.core.waveglow.synthesizer import Synthesizer as WGSynthesizer


class Synthesizer():
  def __init__(self, tacotron_path: str, waveglow_path: str, custom_taco_hparams: Optional[str], custom_wg_hparams: Optional[str], logger: logging.Logger):
    super().__init__()
    self._logger = logger

    self.taco_synt = TacoSynthesizer(
      checkpoint_path=tacotron_path,
      custom_hparams=custom_taco_hparams,
      logger=logger
    )

    self._wg_synt = WGSynthesizer(
      checkpoint_path=waveglow_path,
      custom_hparams=custom_wg_hparams,
      logger=logger
    )

  def infer(self, symbol_ids: List[int], accent_ids: List[int], speaker_id: int, sigma: float, denoiser_strength: float) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], np.ndarray]:
    mel_outputs, mel_outputs_postnet, alignments = self.taco_synt.infer(
      symbol_ids=symbol_ids,
      accent_ids=accent_ids,
      speaker_id=speaker_id
    )

    synthesized_sentence = self._wg_synt.infer(
      mel=mel_outputs_postnet,
      sigma=sigma,
      denoiser_strength=denoiser_strength
    )

    return ((mel_outputs, mel_outputs_postnet, alignments), synthesized_sentence)
