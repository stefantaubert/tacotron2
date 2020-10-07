import os
from logging import Logger
from typing import Dict, Optional

from src.app.io import (get_checkpoints_dir, get_train_log_file,
                        get_train_logs_dir, load_trainset, load_valset,
                        save_prep_name, save_testset, save_trainset,
                        save_valset)
from src.app.pre.prepare import get_prepared_dir, load_filelist
from src.app.utils import prepare_logger
from src.app.waveglow.io import get_train_dir
from src.core.common.train import get_custom_or_last_checkpoint
from src.core.pre.merge_ds import split_prepared_data_train_test_val
from src.core.waveglow.model_checkpoint import CheckpointWaveglow
from src.core.waveglow.train import continue_train, train


def try_load_checkpoint(base_dir: str, train_name: Optional[str], checkpoint: Optional[int], logger: Logger) -> Optional[CheckpointWaveglow]:
  result = None
  if train_name:
    train_dir = get_train_dir(base_dir, train_name, False)
    checkpoint_path, _ = get_custom_or_last_checkpoint(
      get_checkpoints_dir(train_dir), checkpoint)
    result = CheckpointWaveglow.load(checkpoint_path, logger)
  return result


def start_new_training(base_dir: str, train_name: str, prep_name: str, test_size: float = 0.01, validation_size: float = 0.01, custom_hparams: Optional[Dict[str, str]] = None, split_seed: int = 1234, warm_start_train_name: Optional[str] = None, warm_start_checkpoint: Optional[int] = None):
  prep_dir = get_prepared_dir(base_dir, prep_name)
  wholeset = load_filelist(prep_dir)
  trainset, testset, valset = split_prepared_data_train_test_val(
    wholeset, test_size=test_size, validation_size=validation_size, seed=split_seed, shuffle=True)
  train_dir = get_train_dir(base_dir, train_name, create=True)
  save_trainset(train_dir, trainset)
  save_testset(train_dir, testset)
  save_valset(train_dir, valset)

  logs_dir = get_train_logs_dir(train_dir)
  logger = prepare_logger(get_train_log_file(logs_dir), reset=True)

  warm_model = try_load_checkpoint(
    base_dir=base_dir,
    train_name=warm_start_train_name,
    checkpoint=warm_start_checkpoint,
    logger=logger
  )

  save_prep_name(train_dir, prep_name)

  train(
    custom_hparams=custom_hparams,
    logdir=logs_dir,
    trainset=trainset,
    valset=valset,
    save_checkpoint_dir=get_checkpoints_dir(train_dir),
    debug_logger=logger,
    warm_model=warm_model,
  )


def continue_training(base_dir: str, train_name: str, custom_hparams: Optional[Dict[str, str]] = None):
  train_dir = get_train_dir(base_dir, train_name, create=False)
  assert os.path.isdir(train_dir)

  logs_dir = get_train_logs_dir(train_dir)
  logger = prepare_logger(get_train_log_file(logs_dir))

  continue_train(
    custom_hparams=custom_hparams,
    logdir=logs_dir,
    trainset=load_trainset(train_dir),
    valset=load_valset(train_dir),
    save_checkpoint_dir=get_checkpoints_dir(train_dir),
    debug_logger=logger
  )


if __name__ == "__main__":
  mode = 0
  if mode == 0:
    start_new_training(
      base_dir="/datasets/models/taco2pt_v5",
      train_name="debug",
      prep_name="thchs_ljs",
      custom_hparams={
        "batch_size": 3,
        "iters_per_checkpoint": 5,
        "cache_wavs": False
      },
      validation_size=0.001,
    )

  elif mode == 1:
    continue_training(
      base_dir="/datasets/models/taco2pt_v5",
      train_name="debug"
    )
