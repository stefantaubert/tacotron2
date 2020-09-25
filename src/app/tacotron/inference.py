import datetime
import os
from typing import List, Optional

import matplotlib.pylab as plt
from tqdm import tqdm

from src.app.io import (get_checkpoints_dir, get_infer_log,
                        get_inference_root_dir, load_prep_name, save_infer_wav)
from src.app.pre.inference import get_infer_sentences
from src.app.tacotron.defaults import (DEFAULT_DENOISER_STRENGTH,
                                       DEFAULT_SAMPLING_RATE,
                                       DEFAULT_SENTENCE_PAUSE_S, DEFAULT_SIGMA,
                                       DEFAULT_WAVEGLOW)
from src.app.tacotron.io import get_train_dir
from src.app.utils import (add_console_out_to_logger, add_file_out_to_logger,
                           get_default_logger, init_logger)
from src.app.waveglow.io import get_train_dir as get_wg_train_dir
from src.core.common.audio import float_to_wav
from src.core.common.mel_plot import plot_melspec
from src.core.common.train import (get_custom_or_last_checkpoint,
                                   get_last_checkpoint)
from src.core.common.utils import (get_parent_dirname, get_subdir, parse_json,
                                   stack_images_horizontally,
                                   stack_images_vertically)
from src.core.inference.infer import infer as infer_core
from src.core.inference.synthesizer import InferenceResult
from src.core.tacotron.training import CheckpointTacotron
from src.core.waveglow.train import CheckpointWaveglow


def get_infer_dir(train_dir: str, input_name: str, iteration: int, speaker_name: str):
  subdir_name = f"{datetime.datetime.now():%Y-%m-%d,%H-%M-%S},text={input_name},speaker={speaker_name},it={iteration}"
  return get_subdir(get_inference_root_dir(train_dir), subdir_name, create=True)


def load_infer_symbols_map(symbols_map: str) -> List[str]:
  return parse_json(symbols_map)


def save_infer_sentence_plot(infer_dir: str, infer_res: InferenceResult):
  plot_melspec(infer_res.mel_outputs, title="{}: {}".format(
    get_parent_dirname(infer_dir), infer_res.sentence.sent_id))
  path = os.path.join(infer_dir, f"{infer_res.sentence.sent_id}.png")
  plt.savefig(path, bbox_inches='tight')


def save_infer_pre_postnet_sentence_plot(infer_dir: str, infer_res: InferenceResult):
  plot_melspec(infer_res.mel_outputs_postnet,
               title="{}: Pre-Postnet {}".format(get_parent_dirname(infer_dir), infer_res.sentence.sent_id))
  path = os.path.join(infer_dir, f"{infer_res.sentence.sent_id}_pre_post.png")
  plt.savefig(path, bbox_inches='tight')


def save_infer_alignments_sentence_plot(infer_dir: str, infer_res: InferenceResult):
  plot_melspec(infer_res.alignments, title="{}: Alignments {}".format(
    get_parent_dirname(infer_dir), infer_res.sentence.sent_id))
  path = os.path.join(infer_dir, f"{infer_res.sentence.sent_id}_alignments.png")
  plt.savefig(path, bbox_inches='tight')


def save_infer_wav_sentence(infer_dir: str, infer_res: InferenceResult):
  path = os.path.join(infer_dir, f"{infer_res.sentence.sent_id}.wav")
  float_to_wav(infer_res.wav, path, sample_rate=infer_res.sampling_rate)


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


def infer(base_dir: str, train_name: str, text_name: str, ds_speaker: str, waveglow: str = DEFAULT_WAVEGLOW, custom_checkpoint: Optional[int] = None, sentence_pause_s: float = DEFAULT_SENTENCE_PAUSE_S, sigma: float = DEFAULT_SIGMA, denoiser_strength: float = DEFAULT_DENOISER_STRENGTH, sampling_rate: float = DEFAULT_SAMPLING_RATE, analysis: bool = True):
  train_dir = get_train_dir(base_dir, train_name, create=False)
  assert os.path.isdir(train_dir)

  logger = get_default_logger()
  init_logger(logger)
  add_console_out_to_logger(logger)

  logger.info("Inferring...")

  checkpoint_path, iteration = get_custom_or_last_checkpoint(
    get_checkpoints_dir(train_dir), custom_checkpoint)
  taco_checkpoint = CheckpointTacotron.load(checkpoint_path, logger)

  prep_name = load_prep_name(train_dir)
  infer_sents = get_infer_sentences(base_dir, prep_name, text_name)

  infer_dir = get_infer_dir(train_dir, text_name, iteration, ds_speaker)
  add_file_out_to_logger(logger, get_infer_log(infer_dir))

  train_dir_wg = get_wg_train_dir(base_dir, waveglow, create=False)
  wg_checkpoint_path, _ = get_last_checkpoint(get_checkpoints_dir(train_dir_wg))
  wg_checkpoint = CheckpointWaveglow.load(wg_checkpoint_path, logger)

  wav, inference_results = infer_core(
    tacotron_checkpoint=taco_checkpoint,
    waveglow_checkpoint=wg_checkpoint,
    ds_speaker=ds_speaker,
    sentence_pause_s=sentence_pause_s,
    sigma=sigma,
    denoiser_strength=denoiser_strength,
    sentences=infer_sents,
    custom_taco_hparams=None,
    custom_wg_hparams=None,
    logger=logger
  )

  logger.info("Saving wav...")
  save_infer_wav(infer_dir, sampling_rate, wav)

  if analysis:
    logger.info("Analysing...")
    infer_res: InferenceResult
    for infer_res in tqdm(inference_results):
      save_infer_wav_sentence(infer_dir, infer_res)
      save_infer_sentence_plot(infer_dir, infer_res)
      save_infer_pre_postnet_sentence_plot(infer_dir, infer_res)
      save_infer_alignments_sentence_plot(infer_dir, infer_res)
    sent_ids = [x.sentence.sent_id for x in inference_results]
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
    ds_speaker="arctic,BWC",
    analysis=True
  )
