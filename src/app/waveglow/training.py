import os
from typing import Optional

from src.app.io import (get_checkpoints_dir, get_train_log_file,
                        get_train_logs_dir, load_settings, load_trainset,
                        load_valset, save_settings, save_testset,
                        save_trainset, save_valset)
from src.app.pre.prepare import get_prepared_dir, load_filelist
from src.app.utils import (add_console_out_to_logger, add_file_out_to_logger,
                           init_logger, reset_file_log)
from src.app.waveglow.io import get_train_dir
from src.core.pre.merge_ds import split_prepared_data_train_test_val
from src.core.waveglow.train import get_logger as get_train_logger
from src.core.waveglow.train import train as train_core


def train(base_dir: str, train_name: str, prep_name: str, test_size: float = 0.01, validation_size: float = 0.01, hparams: Optional[str] = None, split_seed: int = 1234):
  prep_dir = get_prepared_dir(base_dir, prep_name)
  wholeset = load_filelist(prep_dir)
  trainset, testset, valset = split_prepared_data_train_test_val(
    wholeset, test_size=test_size, validation_size=validation_size, seed=split_seed, shuffle=True)
  train_dir = get_train_dir(base_dir, train_name, create=True)
  save_trainset(train_dir, trainset)
  save_testset(train_dir, testset)
  save_valset(train_dir, valset)

  init_logger(get_train_logger())
  add_console_out_to_logger(get_train_logger())
  logs_dir = get_train_logs_dir(train_dir)
  log_file = get_train_log_file(logs_dir)
  reset_file_log(log_file)
  add_file_out_to_logger(get_train_logger(), log_file)
  # todo log map & args

  save_settings(train_dir, prep_name, hparams)

  train_core(
    custom_hparams=hparams,
    logdir=logs_dir,
    trainset=trainset,
    valset=valset,
    save_checkpoint_dir=get_checkpoints_dir(train_dir),
    continue_train=False
  )


def continue_train(base_dir: str, train_name: str, hparams: Optional[str] = None):
  train_dir = get_train_dir(base_dir, train_name, create=False)
  assert os.path.isdir(train_dir)

  init_logger(get_train_logger())
  add_console_out_to_logger(get_train_logger())
  logs_dir = get_train_logs_dir(train_dir)
  log_file = get_train_log_file(logs_dir)
  add_file_out_to_logger(get_train_logger(), log_file)

  _, custom_hparams_loaded = load_settings(train_dir)

  train_core(
    custom_hparams=hparams if hparams is not None else custom_hparams_loaded,
    logdir=logs_dir,
    trainset=load_trainset(train_dir),
    valset=load_valset(train_dir),
    save_checkpoint_dir=get_checkpoints_dir(train_dir),
    continue_train=True
  )


if __name__ == "__main__":
  mode = 1
  if mode == 1:
    train(
      base_dir="/datasets/models/taco2pt_v5",
      train_name="debug",
      prep_name="thchs_ljs",
      hparams="batch_size=4,iters_per_checkpoint=50,cache_wavs=False"
    )
  elif mode == 2:
    continue_train(
      base_dir="/datasets/models/taco2pt_v5",
      train_name="debug",
      hparams="batch_size=4,iters_per_checkpoint=50,cache_wavs=False"
    )
