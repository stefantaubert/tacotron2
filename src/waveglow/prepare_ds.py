import argparse
import os
from collections import OrderedDict
from math import sqrt
from shutil import copyfile
from tqdm import tqdm

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from src.tacotron.hparams import create_hparams
from src.paths import (ds_preprocessed_file_name, ds_preprocessed_symbols_name,
                   filelist_file_log_name, filelist_file_name,
                   filelist_symbols_file_name, filelist_weights_file_name,
                   get_all_speakers_path, get_ds_dir, get_filelist_dir,
                   train_map_file, filelist_speakers_name)
from src.text.symbol_converter import (deserialize_symbol_ids, init_from_symbols,
                                   load_from_file, serialize_symbol_ids)
from torch import nn
from src.common.train_log import log
from src.common.utils import (csv_separator, duration_col, parse_ds_speakers, parse_json,
                   serialize_ds_speaker, serialize_ds_speakers, speaker_id_col,
                   speaker_name_col, symbols_str_col, utt_name_col,
                   wavpath_col, save_json)

wav_path_col = 0
duration_col = 1

def load_filepaths(filename):
  data = pd.read_csv(filename, header=None, sep=csv_separator)
  data = data.iloc[:, [wav_path_col]]
  data = data.values
  return data

def prepare(base_dir: str, training_dir_path: str, speakers: str):
  ds_speakers = parse_ds_speakers(speakers)
  
  # expand all
  expanded_speakers = []
  for ds, speaker, _ in ds_speakers:
    if speaker == 'all':
      all_speakers_path = get_all_speakers_path(base_dir, ds)
      all_speakers = parse_json(all_speakers_path)
      all_speakers = sorted(all_speakers.keys())
      for speaker_name in all_speakers:
        expanded_speakers.append((ds, speaker_name))
    else:
      expanded_speakers.append((ds, speaker))
  
  expanded_speakers = list(sorted(set(expanded_speakers)))
  new_speakers_str = serialize_ds_speakers(expanded_speakers)
  ds_speakers = parse_ds_speakers(new_speakers_str)
  speakers_info = OrderedDict([(serialize_ds_speaker(ds, speaker), s_id) for ds, speaker, s_id in ds_speakers])
  speakers_file = os.path.join(get_filelist_dir(training_dir_path), filelist_speakers_name)
  save_json(speakers_file, speakers_info)
  print(speakers_info)

  result = []

  for ds, speaker, speaker_id in tqdm(ds_speakers):
    speaker_dir_path = get_ds_dir(base_dir, ds, speaker)
    prepr_path = os.path.join(speaker_dir_path, ds_preprocessed_file_name)
    speaker_data = pd.read_csv(prepr_path, header=None, sep=csv_separator)

    for _, row in speaker_data.iterrows():
      wav_path = row[1]
      duration = row[3]
      new_row = [''] * 2
      new_row[wav_path_col] = wav_path
      new_row[duration_col] = duration
      result.append(new_row)

  # filelist.csv
  df = pd.DataFrame(result)
  print(df.head())
  df.to_csv(os.path.join(get_filelist_dir(training_dir_path), filelist_file_name), header=None, index=None, sep=csv_separator)

  log(training_dir_path, "Done.")
