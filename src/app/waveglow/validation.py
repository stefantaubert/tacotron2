import os
from typing import Optional

from src.app.io import (get_checkpoints_dir, get_val_dir, get_val_log,
                        load_testset, load_valset, save_val_orig_plot,
                        save_val_orig_wav, save_val_plot, save_val_wav)
from src.app.utils import prepare_logger
from src.app.waveglow.io import get_train_dir, save_diff_plot, save_v
from src.core.common.train import get_custom_or_last_checkpoint
from src.core.waveglow.inference import infer
from src.core.waveglow.train import CheckpointWaveglow


def validate(base_dir: str, train_name: str, entry_id: Optional[int] = None, ds_speaker: Optional[str] = None, ds: str = "val", custom_checkpoint: int = 0, sigma: float = 0.666, denoiser_strength: float = 0.00):
  train_dir = get_train_dir(base_dir, train_name, create=False)
  assert os.path.isdir(train_dir)

  if ds == "val":
    data = load_valset(train_dir)
  elif ds == "test":
    data = load_testset(train_dir)
  else:
    assert False

  entry = data.get_for_validation(entry_id, ds_speaker)

  checkpoint_path, iteration = get_custom_or_last_checkpoint(
    get_checkpoints_dir(train_dir), custom_checkpoint)
  val_dir = get_val_dir(train_dir, entry, iteration)

  logger = prepare_logger(get_val_log(val_dir))
  logger.info(f"Validating {entry.wav_path}...")

  checkpoint = CheckpointWaveglow.load(checkpoint_path, logger)

  wav, wav_sr, wav_mel, orig_mel = infer(
    wav_path=entry.wav_path,
    denoiser_strength=denoiser_strength,
    sigma=sigma,
    checkpoint=checkpoint,
    custom_hparams=None,
    logger=logger
  )

  save_val_wav(val_dir, wav_sr, wav)
  save_val_plot(val_dir, wav_mel)
  save_val_orig_wav(val_dir, entry.wav_path)
  save_val_orig_plot(val_dir, orig_mel)
  score = save_diff_plot(val_dir)
  save_v(val_dir)

  logger.info(f"Imagescore: {score*100}%")
  logger.info(f"Saved output to: {val_dir}")


if __name__ == "__main__":

  # validate(
  #   base_dir="/datasets/models/taco2pt_v5",
  #   train_name="pretrained",
  #   entry_id=865
  # )

  # validate(
  #   base_dir="/datasets/models/taco2pt_v5",
  #   train_name="pretrained_v2",
  #   entry_id=865
  # )

  validate(
    base_dir="/datasets/models/taco2pt_v5",
    train_name="pretrained_v3",
    entry_id=865
  )
