import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt

from src.app.utils import add_console_out_to_logger, add_file_out_to_logger, init_logger
from src.app.io import (get_checkpoints_dir, load_speakers_json,
                     get_val_dir, get_val_log, load_valset, save_infer_wav,
                     save_val_comparison, save_val_orig_plot, load_testset,
                     save_val_orig_wav, save_val_plot, save_val_wav)
from src.app.tacotron.io import get_train_dir
from src.app.tacotron.training import load_symbol_converter
from src.app.waveglow import get_train_dir as get_wg_train_dir
from src.core.common import (get_custom_or_last_checkpoint,
                             get_last_checkpoint, get_parent_dirname,
                             plot_melspec)
from src.core.inference import get_logger
from src.core.inference import validate as validate_core
from typing import Optional, Tuple

def save_val_pre_postnet_plot(val_dir: str, mel):
  parent_dir = get_parent_dirname(val_dir)
  plot_melspec(mel, title=f"{parent_dir}: Pre-Postnet")
  path = os.path.join(val_dir, f"{parent_dir}_pre_post.png")
  plt.savefig(path, bbox_inches='tight')

def save_val_alignments_sentence_plot(val_dir: str, mel):
  parent_dir = get_parent_dirname(val_dir)
  plot_melspec(mel, title=f"{parent_dir}: Alignments")
  path = os.path.join(val_dir, f"{parent_dir}_alignments.png")
  plt.savefig(path, bbox_inches='tight')


def validate(base_dir: str, train_name: str, waveglow: str, entry_id: Optional[int] = None, ds_speaker: Optional[str] = None, ds: str = "val", custom_checkpoint: Optional[int] = 0, sigma: float = 0.666, denoiser_strength: float = 0.01, sampling_rate: float = 22050):
  train_dir = get_train_dir(base_dir, train_name, create=False)
  assert os.path.isdir(train_dir)

  logger = get_logger()
  init_logger(logger)
  add_console_out_to_logger(logger)
  
  assert ds != ""
  if ds == "val":
    data = load_valset(train_dir)
  elif ds == "test":
    data = load_testset(train_dir)
  else:
    assert False

  speakers = load_speakers_json(train_dir)

  if entry_id:
    entry = data.get_entry(entry_id)
  elif ds_speaker:
    speaker_id = speakers[ds_speaker]
    entry = data.get_random_entry_ds_speaker(speaker_id)
  else:
    entry = data.get_random_entry()
  
  checkpoint_path, iteration = get_custom_or_last_checkpoint(get_checkpoints_dir(train_dir), custom_checkpoint)
  val_dir = get_val_dir(train_dir, entry, iteration)
  add_file_out_to_logger(logger, get_val_log(val_dir))

  train_dir_wg = get_wg_train_dir(base_dir, waveglow, create=False)
  assert os.path.isdir(train_dir_wg)
  wg_checkpoint_path, _ = get_last_checkpoint(get_checkpoints_dir(train_dir_wg))

  wav, mel_outputs, mel_outputs_postnet, alignments, orig_mel = validate_core(
    entry=entry,
    taco_path=checkpoint_path,
    waveglow_path=wg_checkpoint_path,
    denoiser_strength=denoiser_strength,
    sigma=sigma,
    conv=load_symbol_converter(train_dir),
    n_speakers=len(speakers)
  )

  save_val_wav(val_dir, sampling_rate, wav)
  save_val_plot(val_dir, mel_outputs)
  save_val_orig_wav(val_dir, entry.wav_path)
  save_val_orig_plot(val_dir, orig_mel)
  save_val_pre_postnet_plot(val_dir, mel_outputs_postnet)
  save_val_alignments_sentence_plot(val_dir, alignments)
  save_val_comparison(val_dir)

  logger.info(f"Saved output to: {val_dir}")

if __name__ == "__main__":
  validate(
    base_dir="/datasets/models/taco2pt_v4",
    train_name="debug",
    waveglow="pretrained",
  )

  validate(
    base_dir="/datasets/models/taco2pt_v4",
    train_name="debug",
    entry_id=6,
    waveglow="pretrained",
  )