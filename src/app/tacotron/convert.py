import os
from typing import Optional

from src.app.io import (get_checkpoints_dir, get_train_checkpoints_log_file,
                        get_train_log_file, get_train_logs_dir, load_settings,
                        load_trainset, load_valset, save_settings,
                        save_testset, save_trainset, save_valset)
from src.app.pre.mapping import try_load_symbols_map
from src.app.pre.prepare import (get_prepared_dir, load_filelist,
                                 load_filelist_accents_ids,
                                 load_filelist_speakers_json,
                                 load_filelist_symbol_converter)
from src.app.tacotron.io import get_train_dir
from src.app.utils import (add_console_out_to_logger, add_file_out_to_logger,
                           init_logger, reset_file_log)
from src.core.common.train import get_custom_or_last_checkpoint
from src.core.pre.merge_ds import split_prepared_data_train_test_val
from src.core.tacotron.hparams import create_hparams
from src.core.tacotron.training import continue_train as continue_train_core
from src.core.tacotron.training import (get_checkpoints_eval_logger,
                                        get_mapped_symbol_weights,
                                        get_train_logger, load_state_dict_from,
                                        load_symbol_embedding_weights_from)
from src.core.tacotron.training import train as train_core

# TODO
def convert(base_dir: str, train_name: str, prep_name: str, warm_start_train_name: Optional[str] = None, warm_start_checkpoint: Optional[int] = None, test_size: float = 0.01, validation_size: float = 0.05, custom_hparams: Optional[str] = None, split_seed: int = 1234, weights_train_name: Optional[str] = None, weights_checkpoint: Optional[int] = None, weights_map: Optional[str] = None):
  prep_dir = get_prepared_dir(base_dir, prep_name)
  wholeset = load_filelist(prep_dir)
  trainset, testset, valset = split_prepared_data_train_test_val(
    wholeset, test_size=test_size, validation_size=validation_size, seed=split_seed, shuffle=True)
  train_dir = get_train_dir(base_dir, train_name, create=True)
  save_trainset(train_dir, trainset)
  save_testset(train_dir, testset)
  save_valset(train_dir, valset)

  init_logger(get_train_logger())
  init_logger(get_checkpoints_eval_logger())
  logs_dir = get_train_logs_dir(train_dir)
  log_file = get_train_log_file(logs_dir)
  checkpoints_log_file = get_train_checkpoints_log_file(logs_dir)
  reset_file_log(log_file)
  reset_file_log(checkpoints_log_file)
  add_console_out_to_logger(get_train_logger())
  add_console_out_to_logger(get_checkpoints_eval_logger())
  add_file_out_to_logger(get_train_logger(), log_file)
  add_file_out_to_logger(get_checkpoints_eval_logger(), checkpoints_log_file)

  save_settings(train_dir, prep_name, custom_hparams)

  symbol_ids = load_filelist_symbol_converter(prep_dir)
  speakers = load_filelist_speakers_json(prep_dir)
  accent_ids = load_filelist_accents_ids(prep_dir)

  hparams = create_hparams(len(speakers), len(symbol_ids), len(accent_ids), custom_hparams)

  if weights_train_name:
    weights_train_dir = get_train_dir(base_dir, weights_train_name, False)
    weights_checkpoint_path, _ = get_custom_or_last_checkpoint(
      get_checkpoints_dir(weights_train_dir), weights_checkpoint)
    weights_prep_name, _ = load_settings(weights_train_dir)
    weights_prep_dir = get_prepared_dir(base_dir, weights_prep_name)
    weights_model_symbols_conv = load_filelist_symbol_converter(weights_prep_dir)
    weights = load_symbol_embedding_weights_from(weights_checkpoint_path)
    weights_map = try_load_symbols_map(weights_map)

    mapped_emb_weights = get_mapped_symbol_weights(
      model_symbols=symbol_ids,
      trained_weights=weights,
      trained_symbols=weights_model_symbols_conv,
      custom_mapping=weights_map,
      hparams=hparams
    )
  else:
    mapped_emb_weights = None

  if warm_start_train_name:
    warm_start_train_dir = get_train_dir(base_dir, warm_start_train_name, False)
    warm_start_model_path, _ = get_custom_or_last_checkpoint(
      get_checkpoints_dir(warm_start_train_dir), warm_start_checkpoint)
    warm_states = load_state_dict_from(warm_start_model_path)
  else:
    warm_states = None

  train_core(
    warm_start_states=warm_states,
    hparams=hparams,
    logdir=logs_dir,
    symbol_ids=symbol_ids,
    speakers=speakers,
    accent_ids=accent_ids,
    trainset=trainset,
    valset=valset,
    save_checkpoint_dir=get_checkpoints_dir(train_dir),
    pretrained_weights=mapped_emb_weights
  )


def continue_train(base_dir: str, train_name: str, custom_hparams: Optional[str] = None):
  train_dir = get_train_dir(base_dir, train_name, create=False)
  assert os.path.isdir(train_dir)

  init_logger(get_train_logger())
  init_logger(get_checkpoints_eval_logger())
  logs_dir = get_train_logs_dir(train_dir)
  log_file = get_train_log_file(logs_dir)
  checkpoints_log_file = get_train_checkpoints_log_file(logs_dir)
  add_console_out_to_logger(get_train_logger())
  add_console_out_to_logger(get_checkpoints_eval_logger())
  add_file_out_to_logger(get_train_logger(), log_file)
  add_file_out_to_logger(get_checkpoints_eval_logger(), checkpoints_log_file)

  continue_train_core(
    custom_hparams=custom_hparams,
    logdir=logs_dir,
    trainset=load_trainset(train_dir),
    valset=load_valset(train_dir),
    save_checkpoint_dir=get_checkpoints_dir(train_dir)
  )


if __name__ == "__main__":
  mode = 1
  if mode == 0:
    continue_train(
      base_dir="/datasets/models/taco2pt_v5",
      train_name="debug",
    )
  elif mode == 1:
    train(
      base_dir="/datasets/models/taco2pt_v5",
      train_name="debug",
      prep_name="thchs_ljs",
      custom_hparams="batch_size=17,iters_per_checkpoint=5,epochs_per_checkpoint=1,accents_use_own_symbols=True"
    )
  elif mode == 2:
    train(
      base_dir="/datasets/models/taco2pt_v5",
      train_name="debug",
      prep_name="thchs_ipa",
      warm_start_train_name="ljs_ipa_scratch",
      weights_train_name="ljs_ipa_scratch",
      custom_hparams="batch_size=17,iters_per_checkpoint=0,epochs_per_checkpoint=1"
    )
  elif mode == 3:
    train(
      base_dir="/datasets/models/taco2pt_v5",
      train_name="debug",
      prep_name="thchs_ipa",
      warm_start_train_name="ljs_ipa_scratch",
      weights_train_name="ljs_ipa_scratch",
      weights_map="maps/weights/thchs_ipa_ljs_ipa.json",
      custom_hparams="batch_size=17,iters_per_checkpoint=0,epochs_per_checkpoint=1"
    )
  elif mode == 4:
    continue_train(
      base_dir="/datasets/models/taco2pt_v5",
      train_name="debug",
      custom_hparams="batch_size=17,iters_per_checkpoint=100,epochs_per_checkpoint=1,cache_mels=True,use_saved_mels=True"
    )
  elif mode == 5:
    train(
      base_dir="/datasets/models/taco2pt_v5",
      train_name="debug",
      prep_name="thchs_ipa_acc",
      warm_start_train_name="ljs_ipa_scratch",
      weights_train_name="ljs_ipa_scratch",
      weights_map="maps/weights/thchs_ipa_acc_ljs_ipa.json",
      custom_hparams="batch_size=17,iters_per_checkpoint=0,epochs_per_checkpoint=1"
    )