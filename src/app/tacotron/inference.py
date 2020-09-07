import datetime
import os
from typing import List, Optional

import matplotlib.pylab as plt
import numpy as np
from tqdm import tqdm

from src.app.io import (get_checkpoints_dir, get_infer_log,
                        get_inference_root_dir, save_infer_wav)
from src.app.pre.inference import get_text_dir, load_inference_csv
from src.app.pre.prepare import (get_prepared_dir, load_filelist_accents_ids,
                                 load_filelist_speakers_json,
                                 load_filelist_symbol_converter)
from src.app.tacotron.io import get_train_dir
from src.app.tacotron.training import load_settings
from src.app.utils import (add_console_out_to_logger, add_file_out_to_logger,
                           init_logger)
from src.app.waveglow.io import get_train_dir as get_wg_train_dir
from src.core.common.audio import float_to_wav
from src.core.common.mel_plot import plot_melspec
from src.core.common.train import (get_custom_or_last_checkpoint,
                                   get_last_checkpoint)
from src.core.common.utils import (get_parent_dirname, get_subdir, parse_json,
                                   stack_images_horizontally,
                                   stack_images_vertically)
from src.core.inference.infer import get_logger
from src.core.inference.infer import infer as infer_core


def get_infer_dir(train_dir: str, input_name: str, iteration: int, speaker_id: int):
  subdir_name = f"{datetime.datetime.now():%Y-%m-%d,%H-%M-%S},text={input_name},speaker={speaker_id},it={iteration}"
  return get_subdir(get_inference_root_dir(train_dir), subdir_name, create=True)


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


def infer(base_dir: str, train_name: str, text_name: str, ds_speaker: str, waveglow: str = "pretrained", custom_checkpoint: Optional[int] = None, sentence_pause_s: float = 0.5, sigma: float = 0.666, denoiser_strength: float = 0.01, sampling_rate: float = 22050, analysis: bool = True):
  train_dir = get_train_dir(base_dir, train_name, create=False)
  assert os.path.isdir(train_dir)

  logger = get_logger()
  init_logger(logger)
  add_console_out_to_logger(logger)

  checkpoint_path, iteration = get_custom_or_last_checkpoint(
    get_checkpoints_dir(train_dir), custom_checkpoint)

  prep_name, custom_taco_hparams_loaded = load_settings(train_dir)
  prep_dir = get_prepared_dir(base_dir, prep_name)
  assert os.path.isdir(prep_dir)

  speakers = load_filelist_speakers_json(prep_dir)
  speaker_id = speakers[ds_speaker]
  infer_dir = get_infer_dir(train_dir, text_name, iteration, speaker_id)
  add_file_out_to_logger(logger, get_infer_log(infer_dir))

  train_dir_wg = get_wg_train_dir(base_dir, waveglow, create=False)
  assert os.path.isdir(train_dir_wg)
  wg_checkpoint_path, _ = get_last_checkpoint(get_checkpoints_dir(train_dir_wg))
  _, custom_wg_hparams_loaded = load_settings(train_dir_wg)

  text_dir = get_text_dir(prep_dir, text_name, create=False)
  assert os.path.isdir(text_dir)
  infer_sents = load_inference_csv(text_dir)

  wav, analysis_stack = infer_core(
    taco_path=checkpoint_path,
    waveglow_path=wg_checkpoint_path,
    symbol_id_dict=load_filelist_symbol_converter(prep_dir),
    accent_id_dict=load_filelist_accents_ids(prep_dir),
    n_speakers=len(speakers),
    speaker_id=speaker_id,
    sentence_pause_s=sentence_pause_s,
    sigma=sigma,
    denoiser_strength=denoiser_strength,
    sampling_rate=sampling_rate,
    sentences=infer_sents,
    custom_taco_hparams=custom_taco_hparams_loaded,
    custom_wg_hparams=custom_wg_hparams_loaded
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
    train_name="debug",
    text_name="north",
    ds_speaker="ljs,1",
    analysis=True
  )
