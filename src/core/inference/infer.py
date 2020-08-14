from src.core.pre import text_to_symbols_pipeline
from src.core.common import Language
from typing import List, Tuple
import logging
from src.core.common import concatenate_audios
from src.core.pre import PreparedData, SymbolConverter

from tqdm import tqdm
import numpy as np
from src.core.inference.synthesizer import Synthesizer

def get_logger():
  return logging.getLogger("infer")

_logger = get_logger()

def infer(taco_path: str, waveglow_path: str, conv: SymbolConverter, n_speakers: int, speaker_id: int, sentence_pause_s: float, sigma: float, denoiser_strength: float, sampling_rate: int, lines: List[str], ipa: bool, ignore_tones: bool, ignore_arcs: bool, subset_id: int, lang: Language, symbols_map: dict):
  _logger.info("Inferring...")

  symbol_ids_sentences = text_to_symbols_pipeline(lines, ipa, ignore_tones, ignore_arcs, subset_id, lang, symbols_map, conv, _logger)

  return _infer_core(taco_path, waveglow_path, conv, n_speakers, speaker_id, sentence_pause_s, sigma, denoiser_strength, sampling_rate, symbol_ids_sentences)

def validate(entry: PreparedData, taco_path: str, waveglow_path: str, denoiser_strength: float, sigma: float, conv: SymbolConverter, n_speakers: int):
  _logger.info("Validating...")

  return _infer_core(taco_path, waveglow_path, conv, n_speakers, entry.speaker_id, 0, sigma, denoiser_strength, 0, entry.serialized_updated_ids)

def _infer_core(taco_path: str, waveglow_path: str, conv: SymbolConverter, n_speakers: int, speaker_id: int, sentence_pause_s: float, sigma: float, denoiser_strength: float, sampling_rate: int, symbol_ids_sentences: List[List[int]]):
  synth = Synthesizer(taco_path, waveglow_path, n_symbols=conv.get_symbol_ids_count(), n_speakers=n_speakers, logger=_logger)
  
  result: List[Tuple] = list
  # Speed is: 1min inference for 3min wav result
  for symbol_ids in tqdm(symbol_ids_sentences):
    _logger.info(f"{conv.ids_to_text(symbol_ids)} ({len(symbol_ids)})")
    inferred = synth.infer(symbol_ids, speaker_id, sigma, denoiser_strength)
    result.append(inferred)

  if len(result) > 1:
    assert sampling_rate
    audios = [wg_out for _, wg_out in result]
    output = concatenate_audios(audios, sentence_pause_s, sampling_rate)
  else:
    output = result[0][1]

  return output, result
