import os

from src.app.utils import add_console_and_file_out_to_logger, reset_log
from src.app.io import (get_checkpoints_dir,
                     get_train_log_file, get_train_logs_dir, load_trainset,
                     load_valset, save_testset, save_trainset,
                     save_valset)
from src.app.pre import get_prepared_dir, load_filelist
from src.app.waveglow.io import get_train_dir
from src.core.pre import split_train_test_val
from src.core.waveglow import get_train_logger
from src.core.waveglow import train as train_core


def train(base_dir: str, train_name: str, fl_name: str, test_size: float = 0.01, validation_size: float = 0.01, hparams = "", split_seed: int = 1234):
  prep_dir = get_prepared_dir(base_dir, fl_name)
  wholeset = load_filelist(prep_dir)
  trainset, testset, valset = split_train_test_val(wholeset, test_size=test_size, val_size=validation_size, seed=split_seed, shuffle=True)
  train_dir = get_train_dir(base_dir, train_name, create=True)
  save_trainset(train_dir, trainset)
  save_testset(train_dir, testset)
  save_valset(train_dir, valset)

  logs_dir = get_train_logs_dir(train_dir)
  log_file = get_train_log_file(logs_dir)
  reset_log(log_file)
  add_console_and_file_out_to_logger(get_train_logger(), log_file)
  # todo log map & args

  train_core(
    custom_hparams=hparams,
    logdir=logs_dir,
    trainset=trainset,
    valset=valset,
    save_checkpoint_dir=get_checkpoints_dir(train_dir),
    continue_train=False
  )

def continue_train(base_dir: str, train_name: str, hparams):
  train_dir = get_train_dir(base_dir, train_name, create=False)
  assert os.path.isdir(train_dir)

  logs_dir = get_train_logs_dir(train_dir)
  log_file = get_train_log_file(logs_dir)
  add_console_and_file_out_to_logger(get_train_logger(), log_file)

  train_core(
    custom_hparams=hparams,
    logdir=logs_dir,
    trainset=load_trainset(train_dir),
    valset=load_valset(train_dir),
    save_checkpoint_dir=get_checkpoints_dir(train_dir),
    continue_train=True
  )

if __name__ == "__main__":
  mode = 2
  if mode == 1:
    train(
      base_dir="/datasets/models/taco2pt_v3",
      train_name="debug",
      fl_name="thchs",
      hparams="batch_size=4,iters_per_checkpoint=50,cache_wavs=False"
    )
  elif mode == 2:
    continue_train(
      base_dir="/datasets/models/taco2pt_v3",
      train_name="debug",
      hparams="batch_size=4,iters_per_checkpoint=50,cache_wavs=False"
    )
