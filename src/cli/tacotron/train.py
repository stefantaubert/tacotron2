import os
from argparse import ArgumentParser

from src.cli.pre import (load_filelist, load_filelist_speakers_json,
                         load_filelist_symbol_converter)
from src.cli.tacotron.paths import (get_checkpoints_dir, get_log_dir,
                                    get_log_file, get_speakers,
                                    get_symbols_conv, get_test_csv,
                                    get_train_csv, get_train_dir, get_val_csv)
from src.core.pre import PreparedDataList, SpeakersIdDict, split_train_test_val
from src.core.tacotron import continue_train, train, Tacotron2Logger, get_train_logger
from src.core.pre import SymbolConverter
import logging
#region IO

def _save_trainset(base_dir: str, train_name: str, dataset: PreparedDataList):
  path = get_train_csv(base_dir, train_name)
  dataset.save(path)

def load_trainset(base_dir: str, train_name: str) -> PreparedDataList:
  path = get_train_csv(base_dir, train_name)
  return PreparedDataList.load(path)

def _save_testset(base_dir: str, train_name: str, dataset: PreparedDataList):
  path = get_test_csv(base_dir, train_name)
  dataset.save(path)

def load_testset(base_dir: str, train_name: str) -> PreparedDataList:
  path = get_test_csv(base_dir, train_name)
  return PreparedDataList.load(path)
  
def _save_valset(base_dir: str, train_name: str, dataset: PreparedDataList):
  path = get_val_csv(base_dir, train_name)
  dataset.save(path)

def load_valset(base_dir: str, train_name: str) -> PreparedDataList:
  path = get_val_csv(base_dir, train_name)
  return PreparedDataList.load(path)
  
def load_speakers_json(base_dir: str, train_name: str) -> SpeakersIdDict:
  speakers_path = get_speakers(base_dir, train_name)
  return SpeakersIdDict.load(speakers_path)
  
def _save_speakers_json(base_dir: str, train_name: str, speakers: SpeakersIdDict):
  speakers_path = get_speakers(base_dir, train_name)
  speakers.save(speakers_path)

def load_symbol_converter(base_dir: str, train_name: str) -> SymbolConverter:
  data_path = get_symbols_conv(base_dir, train_name)
  return SymbolConverter.load_from_file(data_path)
  
def _save_symbol_converter(base_dir: str, train_name: str, data: SymbolConverter):
  data_path = get_symbols_conv(base_dir, train_name)
  data.dump(data_path)

#endregion

def _init_train_logger(base_dir: str, train_name: str):
  # logging.basicConfig(
  #   format='[%(asctime)s] (%(levelname)s) %(message)s',
  #   datefmt='%Y/%m/%d %H:%M:%S',
  #   level=logging.CRITICAL,
  #   #stream=logging.StreamHandler(),
  # )
  #root = logging.getLogger()
  #root.disabled = True
  logger = get_train_logger()
  logger.propagate = False
  logger.setLevel(logging.DEBUG)
  formatter = logging.Formatter(
    '[%(asctime)s] (%(levelname)s) %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S'
  )

  console_handler = logging.StreamHandler()
  console_handler.setLevel(logging.DEBUG)
  console_handler.setFormatter(formatter)
  logger.addHandler(console_handler)
  logger.info("init console logger")

  path = get_log_file(base_dir, train_name)
  fh = logging.FileHandler(path)
  fh.setLevel(logging.DEBUG)
  fh.setFormatter(formatter)
  logger.addHandler(fh)
  logger.info("init fh logger")

def _reset_log(base_dir: str, train_name: str):
  log_file = get_log_file(base_dir, train_name)
  if os.path.isfile(log_file):
    os.remove(log_file)

def init_train_parser(parser: ArgumentParser):
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--training_dir', type=str, required=True)
  parser.add_argument('--warm_start_model', type=str)
  parser.add_argument('--fl_name', type=str)
  parser.add_argument('--test_size', type=float, default=0.001)
  parser.add_argument('--validation_size', type=float, default=0.1)
  parser.add_argument('--split_seed', type=int, default=1234)
  parser.add_argument('--hparams', type=str)
  parser.add_argument('--weight_map_mode', type=str, choices=['same_symbols_only', 'use_map'])
  parser.add_argument('--weight_map', type=str)
  parser.add_argument('--weight_map_model', type=str)
  parser.add_argument('--weight_map_model_symbols', type=str)
  return _train

def _train(base_dir: str, train_name: str, fl_name: str, warm_start_model: str = "", test_size: float = 0.01, validation_size: float = 0.05, hparams = "", split_seed: int = 1234, weight_map_model: str = "", weight_map_model_symbols: str = "", weight_map_mode: str = "", weight_map: str = ""):
  wholeset = load_filelist(base_dir, fl_name)
  trainset, testset, valset = split_train_test_val(wholeset, test_size=test_size, val_size=validation_size, seed=split_seed, shuffle=True)
  _save_trainset(base_dir, train_name, trainset)
  _save_testset(base_dir, train_name, testset)
  _save_valset(base_dir, train_name, valset)

  symbols_conv = load_filelist_symbol_converter(base_dir, fl_name)
  _save_symbol_converter(base_dir, train_name, symbols_conv)

  speakers = load_filelist_speakers_json(base_dir, fl_name)
  _save_speakers_json(base_dir, train_name, speakers)

  #train_logger = init_train_logger(logging.getLogger("taco-train"))
  #add_file_log_train(train_logger, get_log_file(base_dir, train_name))
  _reset_log(base_dir, train_name)
  _init_train_logger(base_dir, train_name)
  # todo delete log file

  train(
    warm_start_model_path=warm_start_model,
    weights_path=weight_map,
    hparams=hparams,
    logdir=get_log_dir(base_dir, train_name),
    n_symbols=symbols_conv.get_symbol_ids_count(),
    n_speakers=len(speakers),
    trainset=trainset,
    valset=valset,
    save_checkpoint_dir=get_checkpoints_dir(base_dir, train_name),
    save_checkpoint_log_dir=get_log_dir(base_dir, train_name)
  )

def init_continue_train_parser(parser: ArgumentParser):
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--training_dir', type=str, required=True)
  parser.add_argument('--hparams', type=str)
  return _continue_train

def _continue_train(base_dir: str, train_name: str, hparams):
  assert os.path.isdir(get_train_dir(base_dir, train_name, create=False))

  symbols_conv = load_symbol_converter(base_dir, train_name)
  speakers = load_speakers_json(base_dir, train_name)
  _init_train_logger(base_dir, train_name)

  continue_train(
    hparams=hparams,
    logdir=get_log_dir(base_dir, train_name),
    n_symbols=symbols_conv.get_symbol_ids_count(),
    n_speakers=len(speakers),
    trainset=load_trainset(base_dir, train_name),
    valset=load_valset(base_dir, train_name),
    save_checkpoint_dir=get_checkpoints_dir(base_dir, train_name),
    save_checkpoint_log_dir=get_log_dir(base_dir, train_name)
  )
  
if __name__ == "__main__":
  _train(
    base_dir="/datasets/models/taco2pt_v2",
    train_name="debug_new",
    fl_name="thchs",
    hparams="batch_size=17,iters_per_checkpoint=5,epochs_per_checkpoint=1,cache_mels=False"
  )

  _continue_train(
    base_dir="/datasets/models/taco2pt_v2",
    train_name="debug_new",
    hparams="batch_size=17,iters_per_checkpoint=5,epochs_per_checkpoint=1,cache_mels=False"
  )
