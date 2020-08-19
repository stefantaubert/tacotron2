import os

from src.app.utils import add_console_and_file_out_to_logger, reset_log
from src.app.io import (get_checkpoints_dir,
                     get_train_log_file, get_train_logs_dir, load_trainset,
                     load_valset, save_testset, save_trainset,
                     save_valset)
from src.app.pre import (get_prepared_dir, load_filelist,
                         load_filelist_speakers_json,
                         load_filelist_symbol_converter)
from src.app.tacotron.io import get_train_dir
from src.core.pre import SpeakersIdDict, SymbolConverter, split_train_test_val
from src.core.tacotron import continue_train as continue_train_core
from src.core.tacotron import get_train_logger
from src.core.tacotron import train as train_core

_speakers_json = "speakers.json"
_symbols_json = "symbols.json"

def load_symbol_converter(train_dir: str) -> SymbolConverter:
  data_path = os.path.join(train_dir, _symbols_json)
  return SymbolConverter.load_from_file(data_path)
  
def save_symbol_converter(train_dir: str, data: SymbolConverter):
  data_path = os.path.join(train_dir, _symbols_json)
  data.dump(data_path)

def load_speakers_json(train_dir: str) -> SpeakersIdDict:
  speakers_path = os.path.join(train_dir, _speakers_json)
  return SpeakersIdDict.load(speakers_path)
  
def save_speakers_json(train_dir: str, speakers: SpeakersIdDict):
  speakers_path = os.path.join(train_dir, _speakers_json)
  speakers.save(speakers_path)

def train(base_dir: str, train_name: str, fl_name: str, warm_start_model: str = "", test_size: float = 0.01, validation_size: float = 0.05, hparams = "", split_seed: int = 1234, weight_map_model: str = "", weight_map_model_symbols: str = "", weight_map_mode: str = "", weight_map: str = ""):
  prep_dir = get_prepared_dir(base_dir, fl_name)
  wholeset = load_filelist(prep_dir)
  trainset, testset, valset = split_train_test_val(wholeset, test_size=test_size, val_size=validation_size, seed=split_seed, shuffle=True)
  train_dir = get_train_dir(base_dir, train_name, create=True)
  save_trainset(train_dir, trainset)
  save_testset(train_dir, testset)
  save_valset(train_dir, valset)

  symbols_conv = load_filelist_symbol_converter(prep_dir)
  save_symbol_converter(train_dir, symbols_conv)

  speakers = load_filelist_speakers_json(prep_dir)
  save_speakers_json(train_dir, speakers)

  logs_dir = get_train_logs_dir(train_dir)
  log_file = get_train_log_file(logs_dir)
  reset_log(log_file)
  add_console_and_file_out_to_logger(get_train_logger(), log_file)
  # todo log map & args

  train_core(
    warm_start_model_path=warm_start_model,
    weights_path=weight_map,
    custom_hparams=hparams,
    logdir=logs_dir,
    n_symbols=symbols_conv.get_symbol_ids_count(),
    n_speakers=len(speakers),
    trainset=trainset,
    valset=valset,
    save_checkpoint_dir=get_checkpoints_dir(train_dir),
    save_checkpoint_log_dir=logs_dir
  )

def continue_train(base_dir: str, train_name: str, hparams):
  train_dir = get_train_dir(base_dir, train_name, create=False)
  assert os.path.isdir(train_dir)

  symbols_conv = load_symbol_converter(train_dir)
  speakers = load_speakers_json(train_dir)

  logs_dir = get_train_logs_dir(train_dir)
  log_file = get_train_log_file(logs_dir)
  add_console_and_file_out_to_logger(get_train_logger(), log_file)

  continue_train_core(
    custom_hparams=hparams,
    logdir=logs_dir,
    n_symbols=symbols_conv.get_symbol_ids_count(),
    n_speakers=len(speakers),
    trainset=load_trainset(train_dir),
    valset=load_valset(train_dir),
    save_checkpoint_dir=get_checkpoints_dir(train_dir),
    save_checkpoint_log_dir=logs_dir
  )


if __name__ == "__main__":
  mode = 2
  if mode == 1:
    train(
      base_dir="/datasets/models/taco2pt_v3",
      train_name="debug",
      fl_name="thchs",
      hparams="batch_size=17,iters_per_checkpoint=5,epochs_per_checkpoint=1,cache_mels=False"
    )
  elif mode == 2:
    continue_train(
      base_dir="/datasets/models/taco2pt_v3",
      train_name="debug",
      hparams="batch_size=17,iters_per_checkpoint=5,epochs_per_checkpoint=1,cache_mels=False"
    )
