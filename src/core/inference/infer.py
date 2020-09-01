#from src.core.pre import text_to_symbols_pipeline
from src.core.common import Language, TacotronSTFT, create_hparams, mel_to_numpy
from typing import List, Tuple
import logging
import torch
from src.core.common import concatenate_audios, SymbolIdDict
from src.core.pre import PreparedData
from tqdm import tqdm
import numpy as np
from src.core.inference.synthesizer import Synthesizer

def get_logger():
  return logging.getLogger("infer")

_logger = get_logger()

def infer(taco_path: str, waveglow_path: str, conv: SymbolIdDict, n_speakers: int, speaker_id: int, sentence_pause_s: float, sigma: float, denoiser_strength: float, sampling_rate: int, lines: List[str], ipa: bool, ignore_tones: bool, ignore_arcs: bool, subset_id: int, lang: Language, symbols_map: dict) -> Tuple[np.ndarray, List[Tuple[int, Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]]]:
  _logger.info("Inferring...")

  symbol_ids_sentences = text_to_symbols_pipeline(lines, ipa, ignore_tones, ignore_arcs, subset_id, lang, symbols_map, conv, _logger)

  return _infer_core(taco_path, waveglow_path, conv, n_speakers, speaker_id, sentence_pause_s, sigma, denoiser_strength, sampling_rate, symbol_ids_sentences)

def validate(entry: PreparedData, taco_path: str, waveglow_path: str, denoiser_strength: float, sigma: float, conv: SymbolIdDict, n_speakers: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  _logger.info("Validating...")

  hp = create_hparams()
  taco_stft = TacotronSTFT.fromhparams(hp)
  orig_mel = taco_stft.get_mel_tensor_from_file(entry.wav_path)
  symbol_ids = SymbolIdDict.deserialize_symbol_ids(entry.serialized_updated_ids)
  symbol_ids_sentences = [symbol_ids]

  output, result = _infer_core(taco_path, waveglow_path, conv, n_speakers, entry.speaker_id, 0, sigma, denoiser_strength, 0, symbol_ids_sentences)
  mel_outputs, mel_outputs_postnet, alignments = result[0][1]
  return output, mel_outputs, mel_outputs_postnet, alignments, orig_mel.numpy()

def _infer_core(taco_path: str, waveglow_path: str, conv: SymbolIdDict, n_speakers: int, speaker_id: int, sentence_pause_s: float, sigma: float, denoiser_strength: float, sampling_rate: int, symbol_ids_sentences: List[List[int]]) -> Tuple[np.ndarray, List[Tuple[int, Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]]]:
  synth = Synthesizer(taco_path, waveglow_path, n_symbols=conv.get_symbols_count(), n_speakers=n_speakers, logger=_logger)
  
  result: List[Tuple[int, Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]] = list()
  # Speed is: 1min inference for 3min wav result
  for i, symbol_ids in enumerate(tqdm(symbol_ids_sentences)):
    sent_id = i + 1
    _logger.info(f"{sent_id}: {conv.get_text(symbol_ids)} ({len(symbol_ids)})")
    mels, wav = synth.infer(symbol_ids, speaker_id, sigma, denoiser_strength)
    mels_np: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = [mel_to_numpy(x) for x in mels]
    result.append((sent_id, mels_np, wav))

  audios = [wg_out for _, _, wg_out in result]

  _logger.info("Concatening audios...")
  output = concatenate_audios(audios, sentence_pause_s, sampling_rate)

  return output, result
