import os
from typing import Dict, Optional

import matplotlib.pylab as plt

from src.app.io import (get_checkpoints_dir, get_val_dir, get_val_log,
                        load_prep_name, load_testset, load_valset,
                        save_val_comparison, save_val_orig_plot,
                        save_val_orig_wav, save_val_plot, save_val_wav)
from src.app.pre.prepare import get_prepared_dir, load_prep_speakers_json
from src.app.tacotron.defaults import (DEFAULT_DENOISER_STRENGTH,
                                       DEFAULT_SIGMA, DEFAULT_WAVEGLOW)
from src.app.tacotron.io import get_train_dir
from src.app.utils import prepare_logger
from src.app.waveglow.training import get_train_dir as get_wg_train_dir
from src.core.common.mel_plot import plot_melspec
from src.core.common.taco_stft import get_mel
from src.core.common.train import (get_custom_or_last_checkpoint,
                                   get_last_checkpoint)
from src.core.common.utils import get_parent_dirname
from src.core.inference.infer import validate
from src.core.tacotron.training import CheckpointTacotron
from src.core.waveglow.train import CheckpointWaveglow


def save_val_pre_postnet_plot(val_dir: str, mel):
  parent_dir = get_parent_dirname(val_dir)
  plot_melspec(mel, title=f"{parent_dir}: Pre-Postnet")
  path = os.path.join(val_dir, f"{parent_dir}_pre_post.png")
  plt.savefig(path, bbox_inches='tight')
  plt.close()


def save_val_alignments_sentence_plot(val_dir: str, mel):
  parent_dir = get_parent_dirname(val_dir)
  plot_melspec(mel, title=f"{parent_dir}: Alignments")
  path = os.path.join(val_dir, f"{parent_dir}_alignments.png")
  plt.savefig(path, bbox_inches='tight')
  plt.close()


def validate_main(base_dir: str, train_name: str, waveglow: str = DEFAULT_WAVEGLOW, entry_id: Optional[int] = None, speaker: Optional[str] = None, ds: str = "val", custom_checkpoint: Optional[int] = None, sigma: float = DEFAULT_SIGMA, denoiser_strength: float = DEFAULT_DENOISER_STRENGTH, custom_tacotron_hparams: Optional[Dict[str, str]] = None, custom_waveglow_hparams: Optional[Dict[str, str]] = None):
  train_dir = get_train_dir(base_dir, train_name, create=False)
  assert os.path.isdir(train_dir)

  if ds == "val":
    data = load_valset(train_dir)
  elif ds == "test":
    data = load_testset(train_dir)
  else:
    assert False

  speaker_id: Optional[int] = None
  if speaker is not None:
    prep_name = load_prep_name(train_dir)
    prep_dir = get_prepared_dir(base_dir, prep_name, create=False)
    speakers = load_prep_speakers_json(prep_dir)
    speaker_id = speakers.get_id(speaker)

  entry = data.get_for_validation(entry_id, speaker_id)

  checkpoint_path, iteration = get_custom_or_last_checkpoint(
    get_checkpoints_dir(train_dir), custom_checkpoint)
  val_dir = get_val_dir(train_dir, entry, iteration)

  logger = prepare_logger(get_val_log(val_dir))
  logger.info("Validating...")

  taco_checkpoint = CheckpointTacotron.load(checkpoint_path, logger)

  train_dir_wg = get_wg_train_dir(base_dir, waveglow, create=False)
  wg_checkpoint_path, _ = get_last_checkpoint(get_checkpoints_dir(train_dir_wg))
  wg_checkpoint = CheckpointWaveglow.load(wg_checkpoint_path, logger)

  result = validate(
    tacotron_checkpoint=taco_checkpoint,
    waveglow_checkpoint=wg_checkpoint,
    sigma=sigma,
    denoiser_strength=denoiser_strength,
    entry=entry,
    logger=logger,
    custom_taco_hparams=custom_tacotron_hparams,
    custom_wg_hparams=custom_waveglow_hparams
  )

  orig_mel = get_mel(entry.wav_path, custom_hparams=custom_waveglow_hparams)
  save_val_orig_wav(val_dir, entry.wav_path)
  save_val_orig_plot(val_dir, orig_mel)

  save_val_wav(val_dir, result.sampling_rate, result.wav)
  save_val_plot(val_dir, result.mel_outputs)
  save_val_pre_postnet_plot(val_dir, result.mel_outputs_postnet)
  save_val_alignments_sentence_plot(val_dir, result.alignments)
  save_val_comparison(val_dir)

  logger.info(f"Saved output to: {val_dir}")


if __name__ == "__main__":
  validate_main(
    base_dir="/datasets/models/taco2pt_v5",
    train_name="ljs_ipa_scratch_128",
  )

  # validate(
  #   base_dir="/datasets/models/taco2pt_v5",
  #   train_name="debug",
  #   entry_id=6,
  #   waveglow="pretrained",
  # )
