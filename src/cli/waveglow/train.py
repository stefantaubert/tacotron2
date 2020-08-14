import os
from argparse import ArgumentParser
from src.core.waveglow import continue_train, train, WaveglowLogger, get_train_logger
from src.cli.waveglow.paths import (get_checkpoints_dir, get_log_dir,
                                    get_log_file, get_speakers,
                                    get_symbols_conv, get_test_csv,
                                    get_train_csv, get_train_dir, get_val_csv)
from src.core.pre import PreparedDataList, SpeakersIdDict, split_train_test_val, SymbolConverter
import logging

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
