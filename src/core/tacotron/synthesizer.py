import logging
import os
from typing import List, Optional

import numpy as np
import torch

from src.core.tacotron.hparams import create_hparams
from src.core.tacotron.training import load_model


class Synthesizer():
  def __init__(self, checkpoint_path: str, n_symbols: int, n_accents: int, n_speakers: int, custom_hparams: Optional[str], logger: logging.Logger):
    super().__init__()
    assert os.path.isfile(checkpoint_path)
    self._logger = logger
    self._logger.info(f"Loading tacotron model from: {checkpoint_path}")
    self._logger.info(f'Loaded {n_symbols} symbols')
    self._logger.info(f'Loaded {n_accents} accents')
    self._logger.info(f'Loaded {n_speakers} speaker(s)')

    # Load model from checkpoint
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model_state_dict = checkpoint_dict['state_dict']

    hparams = create_hparams(n_speakers, n_symbols, n_accents, custom_hparams)

    model = load_model(hparams, logger)
    model.load_state_dict(model_state_dict)

    model = model.cuda()
    model = model.eval()

    self.hparams = hparams
    self.model = model

  def infer(self, symbol_ids: List[int], accent_ids: List[int], speaker_id: int):
    symbols_tensor = np.array([symbol_ids])
    symbols_tensor = torch.from_numpy(symbols_tensor)
    symbols_tensor = torch.autograd.Variable(symbols_tensor)
    symbols_tensor = symbols_tensor.cuda()
    symbols_tensor = symbols_tensor.long()

    accents_tensor = np.array([accent_ids])
    accents_tensor = torch.from_numpy(accents_tensor)
    accents_tensor = torch.autograd.Variable(accents_tensor)
    accents_tensor = accents_tensor.cuda()
    accents_tensor = accents_tensor.long()

    speaker_tensor = torch.IntTensor([speaker_id])
    speaker_tensor = speaker_tensor.cuda()
    speaker_tensor = speaker_tensor.long()

    with torch.no_grad():
      mel_outputs, mel_outputs_postnet, _, alignments = self.model.inference(
        inputs=symbols_tensor,
        accents=accents_tensor,
        speaker_id=speaker_tensor
      )

    return mel_outputs, mel_outputs_postnet, alignments
