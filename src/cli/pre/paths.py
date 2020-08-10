import os
from src.common.utils import get_subdir

pre_dir = 'pre'
pre_data_file = 'data.csv'
pre_speakers_file = 'speakers.json'

def get_pre_ds_path(base_dir: str, ds_name: str, create: bool = False):
  return get_subdir(base_dir, os.path.join(pre_dir, ds_name), create)

def get_pre_ds_data_file(base_dir: str, ds_name: str, create: bool = False):
  return os.path.join(get_pre_ds_path(base_dir, ds_name, create), pre_data_file)

def get_pre_ds_speakers_file(base_dir: str, ds_name: str, create: bool = False):
  return os.path.join(get_pre_ds_path(base_dir, ds_name, create), pre_speakers_file)
