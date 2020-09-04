from typing import List, Tuple

from src.core.waveglow import Synthesizer as WGSynthesizer
from src.core.tacotron import Synthesizer as TacoSynthesizer
import logging
import torch
import numpy as np

class Synthesizer():
  def __init__(self, tacotron_path: str, waveglow_path: str, n_symbols: int, n_speakers: int, custom_taco_hparams: str = "", custom_wg_hparams: str = "", logger: logging.Logger = logging.getLogger()):
    super().__init__()
    self._logger = logger
    self._taco_synt = TacoSynthesizer(
      checkpoint_path=tacotron_path,
      n_speakers=n_speakers,
      n_symbols=n_symbols,
      custom_hparams=custom_taco_hparams,
      logger=logger
    )
    self._wg_synt = WGSynthesizer(
      checkpoint_path=waveglow_path,
      custom_hparams=custom_wg_hparams,
      logger=logger
    )

  def infer(self, symbol_ids: List[int], speaker_id: int, sigma: float, denoiser_strength: float) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], np.ndarray]:
    mel_outputs, mel_outputs_postnet, alignments = self._taco_synt.infer(
      symbol_ids=symbol_ids,
      speaker_id=speaker_id
    )

    synthesized_sentence = self._wg_synt.infer_mel(
      mel=mel_outputs_postnet,
      sigma=sigma,
      denoiser_strength=denoiser_strength
    )

    return ((mel_outputs, mel_outputs_postnet, alignments), synthesized_sentence)
