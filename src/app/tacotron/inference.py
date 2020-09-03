import datetime
import os
from typing import List, Optional, Tuple

import matplotlib.pylab as plt
import numpy as np
from tqdm import tqdm

from src.app.utils import add_console_out_to_logger, add_file_out_to_logger, reset_file_log, init_logger
from src.app.io import (get_checkpoints_dir, get_infer_log, save_infer_plot,
                        load_speakers_json, save_infer_wav, get_inference_root_dir)
from src.app.pre import (load_filelist, load_filelist_speakers_json,
                         load_filelist_symbol_converter)
from src.app.tacotron.io import get_train_dir
from src.app.tacotron.training import load_symbol_converter
from src.app.waveglow import get_train_dir as get_wg_train_dir
from src.core.common import (Language, float_to_wav, get_basename,
                             get_custom_or_last_checkpoint,
                             get_last_checkpoint, get_parent_dirname,
                             get_subdir, parse_json, plot_melspec,
                             stack_images_vertically, stack_images_horizontally)
from src.core.inference import get_logger
from src.core.inference import infer as infer_core


def get_infer_dir(train_dir: str, input_name: str, iteration: int, speaker_id: int):
  subdir_name = f"{datetime.datetime.now():%Y-%m-%d,%H-%M-%S},text={input_name},speaker={speaker_id},it={iteration}"
  return get_subdir(get_inference_root_dir(train_dir), subdir_name, create=True)


def load_infer_text(file_name: str) -> List[str]:
  with open(file_name, "r") as f:
    return f.readlines()


def load_infer_symbols_map(symbols_map: str) -> List[str]:
  return parse_json(symbols_map)


def save_infer_sentence_plot(infer_dir: str, sentence_nr: int, mel: np.ndarray):
  plot_melspec(mel, title="{}: {}".format(get_parent_dirname(infer_dir), sentence_nr))
  path = os.path.join(infer_dir, f"{sentence_nr}.png")
  plt.savefig(path, bbox_inches='tight')


def save_infer_pre_postnet_sentence_plot(infer_dir: str, sentence_nr: int, mel: np.ndarray):
  plot_melspec(mel, title="{}: Pre-Postnet {}".format(get_parent_dirname(infer_dir), sentence_nr))
  path = os.path.join(infer_dir, f"{sentence_nr}_pre_post.png")
  plt.savefig(path, bbox_inches='tight')


def save_infer_alignments_sentence_plot(infer_dir: str, sentence_nr: int, mel: np.ndarray):
  plot_melspec(mel, title="{}: Alignments {}".format(get_parent_dirname(infer_dir), sentence_nr))
  path = os.path.join(infer_dir, f"{sentence_nr}_alignments.png")
  plt.savefig(path, bbox_inches='tight')


def save_infer_wav_sentence(infer_dir: str, sentence_nr: int, sampling_rate: int, sent_wav: np.ndarray):
  path = os.path.join(infer_dir, f"{sentence_nr}.wav")
  float_to_wav(sent_wav, path, normalize=True, sample_rate=sampling_rate)


def save_infer_v_pre_post(infer_dir: str, sentence_ids: List[int]):
  paths = []
  for x in sentence_ids:
    paths.append(os.path.join(infer_dir, f"{x}_pre_post.png"))
  path = os.path.join(infer_dir, f"{get_parent_dirname(infer_dir)}_v_pre_post.png")
  stack_images_vertically(paths, path)


def save_infer_v_alignments(infer_dir: str, sentence_ids: List[int]):
  paths = []
  for x in sentence_ids:
    paths.append(os.path.join(infer_dir, f"{x}_alignments.png"))
  path = os.path.join(infer_dir, f"{get_parent_dirname(infer_dir)}_v_alignments.png")
  stack_images_vertically(paths, path)


def save_infer_v_plot(infer_dir: str, sentence_ids: List[int]):
  paths = []
  for x in sentence_ids:
    paths.append(os.path.join(infer_dir, f"{x}.png"))
  path = os.path.join(infer_dir, f"{get_parent_dirname(infer_dir)}_v.png")
  stack_images_vertically(paths, path)


def save_infer_h_plot(infer_dir: str, sentence_ids: List[int]):
  paths = []
  for x in sentence_ids:
    paths.append(os.path.join(infer_dir, f"{x}.png"))
  path = os.path.join(infer_dir, f"{get_parent_dirname(infer_dir)}_h.png")
  stack_images_horizontally(paths, path)


def infer(base_dir: str, train_name: str, text: str, lang: Language, ds_speaker: str, waveglow: str = "pretrained", ignore_tones: bool = False, ignore_arcs: bool = True, symbols_map: Optional[str] = None, hparams: Optional[str] = None, custom_checkpoint: Optional[int] = None, sentence_pause_s: float = 0.5, sigma: float = 0.666, denoiser_strength: float = 0.01, sampling_rate: float = 22050, analysis: bool = True, ipa: bool = True):
  train_dir = get_train_dir(base_dir, train_name, create=False)
  assert os.path.isdir(train_dir)

  logger = get_logger()
  init_logger(logger)
  add_console_out_to_logger(logger)

  input_name = get_basename(text)
  checkpoint_path, iteration = get_custom_or_last_checkpoint(
    get_checkpoints_dir(train_dir), custom_checkpoint)

  speakers = load_speakers_json(train_dir)
  speaker_id = speakers[ds_speaker]
  accent_id = speakers["thchs,D21"]
  logger.info(f"Speaker id {speaker_id}, accent_id {accent_id}.")
  infer_dir = get_infer_dir(train_dir, input_name, iteration, speaker_id)
  add_file_out_to_logger(logger, get_infer_log(infer_dir))

  train_dir_wg = get_wg_train_dir(base_dir, waveglow, create=False)
  assert os.path.isdir(train_dir_wg)
  wg_checkpoint_path, _ = get_last_checkpoint(get_checkpoints_dir(train_dir_wg))

  wav, analysis_stack = infer_core(
    taco_path=checkpoint_path,
    waveglow_path=wg_checkpoint_path,
    conv=load_symbol_converter(train_dir),
    lines=load_infer_text(text),
    n_speakers=len(speakers),
    speaker_id=speaker_id,
    sentence_pause_s=sentence_pause_s,
    sigma=sigma,
    denoiser_strength=denoiser_strength,
    sampling_rate=sampling_rate,
    ipa=ipa,
    ignore_tones=ignore_tones,
    ignore_arcs=ignore_arcs,
    subset_id=accent_id,
    lang=lang,
    symbols_map=load_infer_symbols_map(symbols_map) if symbols_map else None
  )

  logger.info("Saving wav...")
  save_infer_wav(infer_dir, sampling_rate, wav)

  if analysis:
    logger.info("Analysing...")
    for sent_id, mels, sent_wav in tqdm(analysis_stack):
      mel_outputs, mel_outputs_postnet, alignments = mels
      save_infer_wav_sentence(infer_dir, sent_id, sampling_rate, sent_wav)
      save_infer_sentence_plot(infer_dir, sent_id, mel_outputs)
      save_infer_pre_postnet_sentence_plot(infer_dir, sent_id, mel_outputs_postnet)
      save_infer_alignments_sentence_plot(infer_dir, sent_id, alignments)
    sent_ids = [x[0] for x in analysis_stack]
    save_infer_v_plot(infer_dir, sent_ids)
    save_infer_h_plot(infer_dir, sent_ids)
    save_infer_v_pre_post(infer_dir, sent_ids)
    save_infer_v_alignments(infer_dir, sent_ids)

  logger.info(f"Saved output to {infer_dir}")


if __name__ == "__main__":
  infer(
    base_dir="/datasets/models/taco2pt_v5",
    train_name="ljs_ipa_scratch",
    text="examples/ipa/north_sven_orig.txt",
    lang=Language.IPA,
    ds_speaker="ljs,1",
    symbols_map="maps/inference/en_ipa.json",
    ipa=True,
    analysis=False
  )
