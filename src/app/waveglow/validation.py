import os

from src.app.utils import add_console_out_to_logger, add_file_out_to_logger, init_logger
from src.app.io import (get_checkpoints_dir, load_speakers_json,
                     get_val_dir, get_val_log, load_valset, load_testset,
                     save_val_comparison, save_val_orig_plot,
                     save_val_orig_wav, save_val_plot, save_val_wav)
from src.app.waveglow.io import get_train_dir, save_diff_plot, save_v
from src.core.common import get_custom_or_last_checkpoint
from src.core.waveglow import get_infer_logger
from src.core.waveglow import validate as validate_core
from typing import Optional, Tuple


def validate(base_dir: str, train_name: str, entry_id: Optional[int] = None, ds_speaker: Optional[str] = None, ds: str = "val", custom_checkpoint: int = 0, sigma: float = 0.666, denoiser_strength: float = 0.01, sampling_rate: float = 22050):
  train_dir = get_train_dir(base_dir, train_name, create=False)
  assert os.path.isdir(train_dir)

  logger = get_infer_logger()
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

  wav, wav_mel, orig_mel = validate_core(
    entry=entry,
    custom_hparams=custom_checkpoint,
    denoiser_strength=denoiser_strength,
    sigma=sigma,
    checkpoint_path=checkpoint_path
  )

  save_val_wav(val_dir, sampling_rate, wav)
  save_val_plot(val_dir, wav_mel)
  save_val_orig_wav(val_dir, entry.wav_path)
  save_val_orig_plot(val_dir, orig_mel)
  score = save_diff_plot(val_dir)
  save_v(val_dir)

  logger.info(f"Imagescore: {score*100}%")
  logger.info(f"Saved output to: {val_dir}")

if __name__ == "__main__":
  validate(
    base_dir="/datasets/models/taco2pt_v4",
    train_name="debug",
  )

  validate(
    base_dir="/datasets/models/taco2pt_v4",
    train_name="debug",
    entry_id=31
  )
