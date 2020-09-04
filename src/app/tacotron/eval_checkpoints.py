import os

from src.app.utils import add_console_out_to_logger, init_logger
from src.app.io import get_checkpoints_dir, load_valset
from src.app.tacotron.io import get_train_dir
from src.core.tacotron import eval_checkpoints as eval_checkpoints_core
from src.core.tacotron import get_train_logger

from src.app.tacotron.training import load_settings


def eval_checkpoints(base_dir: str, train_name: str, custom_hparams: str, select: int, min_it: int, max_it: int):
  train_dir = get_train_dir(base_dir, train_name, create=False)
  assert os.path.isdir(train_dir)

  symbols_conv = load_symbol_converter(train_dir)
  speakers = load_speakers_json(train_dir)

  init_logger(get_train_logger())
  add_console_out_to_logger(get_train_logger())

  eval_checkpoints_core(
    custom_hparams=custom_hparams,
    checkpoint_dir=get_checkpoints_dir(train_dir),
    select=select,
    min_it=min_it,
    max_it=max_it,
    n_symbols=len(symbols_conv),
    n_speakers=len(speakers),
    valset=load_valset(train_dir)
  )


if __name__ == "__main__":
  eval_checkpoints(
    base_dir="/datasets/models/taco2pt_v5",
    train_name="debug",
    custom_hparams="batch_size=17",
    select=1,
    min_it=0,
    max_it=0
  )
