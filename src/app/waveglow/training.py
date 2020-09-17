import os
from typing import Optional

from src.app.io import (get_checkpoints_dir, get_train_log_file,
                        get_train_logs_dir, load_trainset, load_valset,
                        save_prep_name, save_testset, save_trainset,
                        save_valset)
from src.app.pre.prepare import get_prepared_dir, load_filelist
from src.app.utils import prepare_logger
from src.app.waveglow.io import get_train_dir
from src.core.pre.merge_ds import split_prepared_data_train_test_val
from src.core.waveglow.train import continue_train as continue_train_core
from src.core.waveglow.train import train as train_core


def train(base_dir: str, train_name: str, prep_name: str, test_size: float = 0.01, validation_size: float = 0.01, custom_hparams: Optional[str] = None, split_seed: int = 1234):
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

  save_prep_name(train_dir, prep_name)

  train_core(
    custom_hparams=custom_hparams,
    logdir=logs_dir,
    trainset=trainset,
    valset=valset,
    save_checkpoint_dir=get_checkpoints_dir(train_dir),
    debug_logger=logger
  )


def continue_train(base_dir: str, train_name: str, custom_hparams: Optional[str] = None):
  train_dir = get_train_dir(base_dir, train_name, create=False)
  assert os.path.isdir(train_dir)

  logs_dir = get_train_logs_dir(train_dir)
  logger = prepare_logger(get_train_log_file(logs_dir))

  continue_train_core(
    custom_hparams=custom_hparams,
    logdir=logs_dir,
    trainset=load_trainset(train_dir),
    valset=load_valset(train_dir),
    save_checkpoint_dir=get_checkpoints_dir(train_dir),
    debug_logger=logger
  )


if __name__ == "__main__":
  mode = 2
  if mode == 1:
    train(
      base_dir="/datasets/models/taco2pt_v5",
      train_name="debug",
      prep_name="thchs_ljs",
      custom_hparams="batch_size=4,iters_per_checkpoint=50,cache_wavs=False"
    )
  elif mode == 2:
    continue_train(
      base_dir="/datasets/models/taco2pt_v5",
      train_name="debug",
      custom_hparams="batch_size=4,iters_per_checkpoint=50,cache_wavs=False"
    )
