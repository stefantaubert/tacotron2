from logging import Logger
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.core.inference.synthesizer import InferenceResult, Synthesizer
from src.core.pre.merge_ds import PreparedData
from src.core.pre.text.pre_inference import InferSentence, InferSentenceList
from src.core.tacotron.training import CheckpointTacotron
from src.core.waveglow.train import CheckpointWaveglow


def validate(tacotron_checkpoint: CheckpointTacotron, waveglow_checkpoint: CheckpointWaveglow, entry: PreparedData, denoiser_strength: float, sigma: float, custom_taco_hparams: Optional[Dict[str, str]], custom_wg_hparams: Optional[Dict[str, str]], logger: Logger) -> InferenceResult:
  model_symbols = tacotron_checkpoint.get_symbols()
  model_accents = tacotron_checkpoint.get_accents()
  model_speakers = tacotron_checkpoint.get_speakers()

  infer_sent = InferSentence(
    sent_id=1,
    symbols=model_symbols.get_symbols(entry.serialized_symbol_ids),
    accents=model_accents.get_accents(entry.serialized_accent_ids)
  )

  _, result = infer(
    tacotron_checkpoint=tacotron_checkpoint,
    waveglow_checkpoint=waveglow_checkpoint,
    ds_speaker=model_speakers.get_speaker(entry.speaker_id),
    sentence_pause_s=0,
    sigma=sigma,
    denoiser_strength=denoiser_strength,
    sentences=InferSentenceList([infer_sent]),
    custom_taco_hparams=custom_taco_hparams,
    custom_wg_hparams=custom_wg_hparams,
    logger=logger
  )

  assert len(result) == 1

  return result[0]


def infer(tacotron_checkpoint: CheckpointTacotron, waveglow_checkpoint: CheckpointWaveglow, ds_speaker: str, sentence_pause_s: float, sigma: float, denoiser_strength: float, sentences: InferSentenceList, custom_taco_hparams: Optional[Dict[str, str]], custom_wg_hparams: Optional[Dict[str, str]], logger: Logger) -> Tuple[np.ndarray, List[InferenceResult]]:
  synth = Synthesizer(
    tacotron_checkpoint,
    waveglow_checkpoint,
    logger=logger,
    custom_taco_hparams=custom_taco_hparams,
    custom_wg_hparams=custom_wg_hparams
  )

  return synth.infer(
    sentences=sentences,
    speaker=ds_speaker,
    denoiser_strength=denoiser_strength,
    sentence_pause_s=sentence_pause_s,
    sigma=sigma,
  )
