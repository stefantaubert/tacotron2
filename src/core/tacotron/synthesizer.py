import logging
import os
from typing import List, Optional

import numpy as np
import torch

from src.core.tacotron.model import get_model_symbol_ids
from src.core.tacotron.training import load_checkpoint, load_model


class Synthesizer():
  def __init__(self, checkpoint_path: str, custom_hparams: Optional[str], logger: logging.Logger):
    super().__init__()
    assert os.path.isfile(checkpoint_path)
    self._logger = logger

    checkpoint = load_checkpoint(checkpoint_path, logger)

    self.accents = checkpoint.accents
    self.symbols = checkpoint.symbols
    self.speakers = checkpoint.speakers

    hparams = checkpoint.hparams
    # todo apply custom_hparams

    model = load_model(
      hparams=hparams,
      state_dict=checkpoint.state_dict,
      logger=logger
    )

    model = model.eval()

    self.hparams = hparams
    self.model = model

  def _get_model_symbols_tensor(self, symbol_ids: List[int], accent_ids: List[int]) -> torch.LongTensor:
    model_symbol_ids = get_model_symbol_ids(
      symbol_ids, accent_ids, self.hparams.n_symbols, self.hparams.accents_use_own_symbols)
    self._logger.debug(f"Symbol ids:\n{symbol_ids}")
    self._logger.debug(f"Model symbol ids:\n{model_symbol_ids}")
    symbols_tensor = np.array([model_symbol_ids])
    symbols_tensor = torch.from_numpy(symbols_tensor)
    symbols_tensor = torch.autograd.Variable(symbols_tensor)
    symbols_tensor = symbols_tensor.cuda()
    symbols_tensor = symbols_tensor.long()
    return symbols_tensor

  def infer(self, symbol_ids: List[int], accent_ids: List[int], speaker_id: int):
    for symbol_id in symbol_ids:
      assert self.symbols.id_exists(symbol_id)
    for accent_id in accent_ids:
      assert self.accents.id_exists(accent_id)
    assert self.speakers.id_exists(speaker_id)

    symbols_tensor = self._get_model_symbols_tensor(symbol_ids, accent_ids)

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
