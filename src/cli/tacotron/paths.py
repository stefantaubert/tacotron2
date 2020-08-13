import os
from src.core.common.utils import get_subdir

_train_dir = "train"
_log_dir = "logs"
_log_file = "log.txt"

_checkpoints_dir = "checkpoints"

_trainset_file = "train.csv"
_testset_file = "test.csv"
_valset_file = "validation.csv"
_symbols_conv = "symbols.json"
_speakers = "speakers.json"

def get_train_dir(base_dir: str, train_name: str, create: bool):
  return get_subdir(base_dir, os.path.join(_train_dir, train_name), create)

def get_train_csv(base_dir: str, train_name: str):
  return os.path.join(get_train_dir(base_dir, train_name, create=True), _trainset_file)

def get_test_csv(base_dir: str, train_name: str):
  return os.path.join(get_train_dir(base_dir, train_name, create=True), _testset_file)

def get_val_csv(base_dir: str, train_name: str):
  return os.path.join(get_train_dir(base_dir, train_name, create=True), _valset_file)

def get_symbols_conv(base_dir: str, train_name: str):
  return os.path.join(get_train_dir(base_dir, train_name, create=True), _symbols_conv)

def get_speakers(base_dir: str, train_name: str):
  return os.path.join(get_train_dir(base_dir, train_name, create=True), _speakers)

def get_log_dir(base_dir: str, train_name: str):
  train_dir = get_train_dir(base_dir, train_name, create=True)
  return get_subdir(train_dir, _log_dir, create=True)

def get_log_file(base_dir: str, train_name: str):
  return os.path.join(get_log_dir(base_dir, train_name), _log_file)

def get_checkpoints_dir(base_dir: str, train_name: str):
  train_dir = get_train_dir(base_dir, train_name, create=True)
  return get_subdir(train_dir, _checkpoints_dir, create=True)
