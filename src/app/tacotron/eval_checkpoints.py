import os

from src.app.io import get_checkpoints_dir, load_prep_name, load_valset
from src.app.pre.prepare import (get_prepared_dir, load_prep_accents_ids,
                                 load_prep_speakers_json,
                                 load_prep_symbol_converter)
from src.app.tacotron.io import get_train_dir
from src.app.utils import prepare_logger
from src.core.tacotron.training import \
    eval_checkpoints as eval_checkpoints_core


def eval_checkpoints_main(base_dir: str, train_name: str, select: int, min_it: int, max_it: int):
  train_dir = get_train_dir(base_dir, train_name, create=False)
  assert os.path.isdir(train_dir)

  prep_name = load_prep_name(train_dir)
  prep_dir = get_prepared_dir(base_dir, prep_name)

  symbols_conv = load_prep_symbol_converter(prep_dir)
  speakers = load_prep_speakers_json(prep_dir)
  accents = load_prep_accents_ids(prep_dir)

  logger = prepare_logger()

  eval_checkpoints_core(
    custom_hparams=None,
    checkpoint_dir=get_checkpoints_dir(train_dir),
    select=select,
    min_it=min_it,
    max_it=max_it,
    n_symbols=len(symbols_conv),
    n_speakers=len(speakers),
    n_accents=len(accents),
    valset=load_valset(train_dir),
    logger=logger
  )


if __name__ == "__main__":
  eval_checkpoints_main(
    base_dir="/datasets/models/taco2pt_v5",
    train_name="debug",
    select=1,
    min_it=0,
    max_it=0
  )
