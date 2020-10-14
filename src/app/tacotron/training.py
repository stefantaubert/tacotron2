import logging
import os
from functools import partial
from logging import Logger
from typing import Dict, Optional

from src.app.io import (get_checkpoints_dir, get_train_checkpoints_log_file,
                        get_train_log_file, get_train_logs_dir, load_prep_name,
                        load_trainset, load_valset, save_prep_name,
                        split_dataset)
from src.app.pre.mapping import load_weights_map
from src.app.pre.prepare import (get_prepared_dir, load_prep_accents_ids,
                                 load_prep_speakers_json,
                                 load_prep_symbol_converter)
from src.app.tacotron.io import get_train_dir
from src.app.utils import prepare_logger
from src.core.common.train import (get_custom_or_last_checkpoint,
                                   get_last_checkpoint, get_pytorch_filename)
from src.core.tacotron.logger import Tacotron2Logger
from src.core.tacotron.training import (CheckpointTacotron, continue_train,
                                        train)


def try_load_checkpoint(base_dir: str, train_name: Optional[str], checkpoint: Optional[int], logger: Logger) -> Optional[CheckpointTacotron]:
  result = None
  if train_name:
    train_dir = get_train_dir(base_dir, train_name, False)
    checkpoint_path, _ = get_custom_or_last_checkpoint(
      get_checkpoints_dir(train_dir), checkpoint)
    result = CheckpointTacotron.load(checkpoint_path, logger)
    logger.info(f"Using warm start model: {checkpoint_path}")
  return result


def save_checkpoint(checkpoint: CheckpointTacotron, save_checkpoint_dir: str, logger: Logger):
  checkpoint_path = os.path.join(
    save_checkpoint_dir, get_pytorch_filename(checkpoint.iteration))
  checkpoint.save(checkpoint_path, logger)


def train_main(base_dir: str, train_name: str, prep_name: str, warm_start_train_name: Optional[str] = None, warm_start_checkpoint: Optional[int] = None, test_size: float = 0.01, validation_size: float = 0.05, custom_hparams: Optional[Dict[str, str]] = None, split_seed: int = 1234, weights_train_name: Optional[str] = None, weights_checkpoint: Optional[int] = None, use_weights_map: Optional[bool] = None, map_from_speaker: Optional[str] = None):
  prep_dir = get_prepared_dir(base_dir, prep_name)
  train_dir = get_train_dir(base_dir, train_name, create=True)
  logs_dir = get_train_logs_dir(train_dir)

  taco_logger = Tacotron2Logger(logs_dir)
  logger = prepare_logger(get_train_log_file(logs_dir), reset=True)
  checkpoint_logger = prepare_logger(
    log_file_path=get_train_checkpoints_log_file(logs_dir),
    logger=logging.getLogger("checkpoint-logger"),
    reset=True
  )

  save_prep_name(train_dir, prep_name)

  trainset, valset = split_dataset(
    prep_dir=prep_dir,
    train_dir=train_dir,
    test_size=test_size,
    validation_size=validation_size,
    split_seed=split_seed
  )

  weights_model = try_load_checkpoint(
    base_dir=base_dir,
    train_name=weights_train_name,
    checkpoint=weights_checkpoint,
    logger=logger
  )

  weights_map = None
  if use_weights_map is not None and use_weights_map:
    weights_train_dir = get_train_dir(base_dir, weights_train_name, False)
    weights_prep_name = load_prep_name(weights_train_dir)
    weights_map = load_weights_map(prep_dir, weights_prep_name)

  warm_model = try_load_checkpoint(
    base_dir=base_dir,
    train_name=warm_start_train_name,
    checkpoint=warm_start_checkpoint,
    logger=logger
  )

  save_callback = partial(
    save_checkpoint,
    save_checkpoint_dir=get_checkpoints_dir(train_dir),
    logger=logger,
  )

  train(
    custom_hparams=custom_hparams,
    taco_logger=taco_logger,
    symbols=load_prep_symbol_converter(prep_dir),
    speakers=load_prep_speakers_json(prep_dir),
    accents=load_prep_accents_ids(prep_dir),
    trainset=trainset,
    valset=valset,
    save_callback=save_callback,
    weights_map=weights_map,
    weights_checkpoint=weights_model,
    warm_model=warm_model,
    map_from_speaker_name=map_from_speaker,
    logger=logger,
    checkpoint_logger=checkpoint_logger,
  )


def continue_train_main(base_dir: str, train_name: str, custom_hparams: Optional[Dict[str, str]] = None):
  train_dir = get_train_dir(base_dir, train_name, create=False)
  assert os.path.isdir(train_dir)

  logs_dir = get_train_logs_dir(train_dir)
  taco_logger = Tacotron2Logger(logs_dir)
  logger = prepare_logger(get_train_log_file(logs_dir))
  checkpoint_logger = prepare_logger(
    log_file_path=get_train_checkpoints_log_file(logs_dir),
    logger=logging.getLogger("checkpoint-logger")
  )

  checkpoints_dir = get_checkpoints_dir(train_dir)
  last_checkpoint_path, _ = get_last_checkpoint(checkpoints_dir)
  last_checkpoint = CheckpointTacotron.load(last_checkpoint_path, logger)

  save_callback = partial(
    save_checkpoint,
    save_checkpoint_dir=checkpoints_dir,
    logger=logger,
  )

  continue_train(
    checkpoint=last_checkpoint,
    custom_hparams=custom_hparams,
    taco_logger=taco_logger,
    trainset=load_trainset(train_dir),
    valset=load_valset(train_dir),
    logger=logger,
    checkpoint_logger=checkpoint_logger,
    save_callback=save_callback
  )


if __name__ == "__main__":
  mode = 2
  if mode == 0:
    train_main(
      base_dir="/datasets/models/taco2pt_v5",
      train_name="debug",
      prep_name="arctic_ipa",
      custom_hparams={
        "batch_size": 17,
        "iters_per_checkpoint": 5,
        "epochs_per_checkpoint": 1,
        "accents_use_own_symbols": False
      }
    )

  elif mode == 1:
    continue_train_main(
      base_dir="/datasets/models/taco2pt_v5",
      train_name="debug",
      custom_hparams={
        "iters_per_checkpoint": 100,
      }
    )

  elif mode == 2:
    train_main(
      base_dir="/datasets/models/taco2pt_v5",
      train_name="debug",
      prep_name="arctic_ipa_22050",
      custom_hparams={
        "batch_size": 26,
        "iters_per_checkpoint": 5,
        "epochs_per_checkpoint": 1,
      },
      weights_train_name="ljs_ipa_scratch_128",
      warm_start_train_name="ljs_ipa_scratch_128",
      map_from_speaker="ljs,1",
      use_weights_map=True,
    )
  elif mode == 3:
    train_main(
      base_dir="/datasets/models/taco2pt_v5",
      train_name="debug",
      prep_name="thchs_ipa",
      warm_start_train_name="ljs_ipa_scratch",
      weights_train_name="ljs_ipa_scratch",
      # weights_map="maps/weights/thchs_ipa_ljs_ipa.json",
      custom_hparams="batch_size=17,iters_per_checkpoint=0,epochs_per_checkpoint=1"
    )
  elif mode == 4:
    continue_train_main(
      base_dir="/datasets/models/taco2pt_v5",
      train_name="debug",
      custom_hparams="batch_size=17,iters_per_checkpoint=100,epochs_per_checkpoint=1,cache_mels=True,use_saved_mels=True"
    )
  elif mode == 5:
    train_main(
      base_dir="/datasets/models/taco2pt_v5",
      train_name="debug",
      prep_name="thchs_ipa_acc",
      warm_start_train_name="ljs_ipa_scratch",
      weights_train_name="ljs_ipa_scratch",
      # weights_map="maps/weights/thchs_ipa_acc_ljs_ipa.json",
      custom_hparams="batch_size=17,iters_per_checkpoint=0,epochs_per_checkpoint=1"
    )
