import numpy as np
import os

import torch
from src.core.tacotron.hparams import create_hparams
from src.core.tacotron.train import load_model
import logging

class Synthesizer():
  def __init__(self, checkpoint_path: str, n_symbols: int, n_speakers: int, custom_hparams: str = "", logger: logging.Logger = logging.getLogger()):
    super().__init__()
    assert os.path.isfile(checkpoint_path)
    self._logger = logger
    self._logger.info(f"Loding tacotron model from: {checkpoint_path}")
    self._logger.info(f'Loaded {n_symbols} symbols')
    self._logger.info(f'Loaded {n_speakers} speaker(s)')

    # Load model from checkpoint
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model_state_dict = checkpoint_dict['state_dict']

    hparams = create_hparams(custom_hparams)
    hparams.n_symbols = n_symbols
    hparams.n_speakers = n_speakers

    model = load_model(hparams)
    model.load_state_dict(model_state_dict)

    model = model.cuda()
    model = model.eval()

    self.hparams = hparams
    self.model = model

  def infer(self, symbol_ids, speaker_id: int):
    sequence = np.array([symbol_ids])
    sequence = torch.from_numpy(sequence)
    sequence = torch.autograd.Variable(sequence)
    sequence = sequence.cuda()
    sequence = sequence.long()

    speaker_id = torch.IntTensor([speaker_id])
    speaker_id = speaker_id.cuda()
    speaker_id = speaker_id.long()

    with torch.no_grad():
      mel_outputs, mel_outputs_postnet, gate_outputs, alignments = self.model.inference(sequence, speaker_id)
    
    return mel_outputs, mel_outputs_postnet, alignments
