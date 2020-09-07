import os
from typing import Optional

from src.app.io import (get_checkpoints_dir, get_val_dir, get_val_log,
                        load_settings, load_testset, load_valset,
                        save_val_orig_plot, save_val_orig_wav, save_val_plot,
                        save_val_wav)
from src.app.pre.prepare import get_prepared_dir, load_filelist_speakers_json
from src.app.utils import (add_console_out_to_logger, add_file_out_to_logger,
                           init_logger)
from src.app.waveglow.io import get_train_dir, save_diff_plot, save_v
from src.core.common.train import get_custom_or_last_checkpoint
from src.core.waveglow.inference import get_logger as get_infer_logger
from src.core.waveglow.inference import validate as validate_core


def validate(base_dir: str, train_name: str, entry_id: Optional[int] = None, ds_speaker: Optional[str] = None, ds: str = "val", custom_checkpoint: int = 0, sigma: float = 0.666, denoiser_strength: float = 0.01, sampling_rate: float = 22050):
  train_dir = get_train_dir(base_dir, train_name, create=False)
  assert os.path.isdir(train_dir)

  logger = get_infer_logger()
  init_logger(logger)
  add_console_out_to_logger(logger)

  if ds == "val":
    data = load_valset(train_dir)
  elif ds == "test":
    data = load_testset(train_dir)
  else:
    assert False

  prep_name, custom_hparams_loaded = load_settings(train_dir)

  if entry_id:
    entry = data.get_entry(entry_id)
  elif ds_speaker:
    prep_dir = get_prepared_dir(base_dir, prep_name)
    assert os.path.isdir(prep_dir)
    speakers = load_filelist_speakers_json(prep_dir)
    speaker_id = speakers[ds_speaker]
    entry = data.get_random_entry_ds_speaker(speaker_id)
  else:
    entry = data.get_random_entry()

  checkpoint_path, iteration = get_custom_or_last_checkpoint(
    get_checkpoints_dir(train_dir), custom_checkpoint)
  val_dir = get_val_dir(train_dir, entry, iteration)
  add_file_out_to_logger(logger, get_val_log(val_dir))

  wav, wav_mel, orig_mel = validate_core(
    entry=entry,
    denoiser_strength=denoiser_strength,
    sigma=sigma,
    checkpoint_path=checkpoint_path,
    custom_hparams=custom_hparams_loaded,
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
    base_dir="/datasets/models/taco2pt_v5",
    train_name="debug",
  )

  validate(
    base_dir="/datasets/models/taco2pt_v5",
    train_name="debug",
    entry_id=31
  )
