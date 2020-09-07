#from src.core.pre import text_to_symbols_pipeline
from src.core.common import AccentsDict
from src.core.common import deserialize_list
from src.core.pre import Sentence
from src.core.pre import InferSentenceList
from src.core.common import Language, TacotronSTFT, create_hparams, mel_to_numpy
from typing import List, Optional, Tuple
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


def infer(taco_path: str, waveglow_path: str, symbol_id_dict: SymbolIdDict, accent_id_dict: AccentsDict, n_speakers: int, speaker_id: int, sentence_pause_s: float, sigma: float, denoiser_strength: float, sampling_rate: int, sentences: InferSentenceList, custom_taco_hparams: Optional[str] = None, custom_wg_hparams: Optional[str] = None) -> Tuple[np.ndarray, List[Tuple[int, Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]]]:
  _logger.info("Inferring...")

  return _infer_core(
    taco_path=taco_path,
    waveglow_path=waveglow_path,
    symbol_id_dict=symbol_id_dict,
    accent_id_dict=accent_id_dict,
    n_speakers=n_speakers,
    speaker_id=speaker_id,
    sentence_pause_s=sentence_pause_s,
    sigma=sigma,
    denoiser_strength=denoiser_strength,
    sampling_rate=sampling_rate,
    sentences=sentences,
    custom_taco_hparams=custom_taco_hparams,
    custom_wg_hparams=custom_wg_hparams
  )


def validate(entry: PreparedData, taco_path: str, waveglow_path: str, denoiser_strength: float, sigma: float, symbol_id_dict: SymbolIdDict, accent_id_dict: AccentsDict, n_speakers: int, custom_taco_hparams: Optional[str] = None, custom_wg_hparams: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  _logger.info("Validating...")

  hparams = create_hparams()
  taco_stft = TacotronSTFT.fromhparams(hparams)
  orig_mel = taco_stft.get_mel_tensor_from_file(entry.wav_path).numpy()

  sentences = InferSentenceList([Sentence(
    sent_id=0,
    text="",
    lang=entry.lang,
    serialized_symbols=entry.serialized_symbol_ids,
    serialized_accents=entry.serialized_accent_ids,
  )])

  output, result = _infer_core(
    taco_path=taco_path,
    waveglow_path=waveglow_path,
    symbol_id_dict=symbol_id_dict,
    accent_id_dict=accent_id_dict,
    n_speakers=n_speakers,
    speaker_id=entry.speaker_id,
    sentence_pause_s=0,
    sigma=sigma,
    denoiser_strength=denoiser_strength,
    sampling_rate=0,
    sentences=sentences,
    custom_taco_hparams=custom_taco_hparams,
    custom_wg_hparams=custom_wg_hparams
  )

  mel_outputs, mel_outputs_postnet, alignments = result[0][1]
  return output, mel_outputs, mel_outputs_postnet, alignments, orig_mel


def _infer_core(taco_path: str, waveglow_path: str, symbol_id_dict: SymbolIdDict, accent_id_dict: AccentsDict, n_speakers: int, speaker_id: int, sentence_pause_s: float, sigma: float, denoiser_strength: float, sampling_rate: int, sentences: InferSentenceList, custom_taco_hparams: Optional[str] = None, custom_wg_hparams: Optional[str] = None) -> Tuple[np.ndarray, List[Tuple[int, Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]]]:
  _logger.debug(f"Selected speaker id: {speaker_id}")

  synth = Synthesizer(
    taco_path,
    waveglow_path,
    n_symbols=len(symbol_id_dict),
    n_accents=len(accent_id_dict),
    n_speakers=n_speakers,
    logger=_logger,
    custom_taco_hparams=custom_taco_hparams,
    custom_wg_hparams=custom_wg_hparams
  )

  result: List[Tuple[int, Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]] = list()

  # Speed is: 1min inference for 3min wav result
  for sentence in sentences.items(True):
    _logger.info(f"\n{sentence.get_formatted(symbol_id_dict, accent_id_dict)}")
    mels, wav = synth.infer(
      symbol_ids=sentence.get_symbol_ids(),
      accent_ids=sentence.get_accent_ids(),
      speaker_id=speaker_id,
      sigma=sigma,
      denoiser_strength=denoiser_strength
    )

    mels_np: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = [mel_to_numpy(x) for x in mels]
    result.append((sentence.sent_id, mels_np, wav))

  audios = [wg_out for _, _, wg_out in result]

  if len(audios) > 0:
    _logger.info("Concatening audios...")
  output = concatenate_audios(audios, sentence_pause_s, sampling_rate)

  return output, result
