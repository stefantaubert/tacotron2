import argparse
import json
import os
import random
from shutil import copyfile

import pandas as pd

from src.common.train_log import reset_log
from src.common.utils import (csv_separator, get_speaker_count_csv,
                              get_utterance_names_csv, parse_ds_speaker,
                              speaker_id_col, speaker_name_col,
                              symbols_str_col, utt_name_col, wavpath_col)
from src.paths import (ds_preprocessed_file_name,
                              ds_preprocessed_symbols_name, filelist_file_name,
                              filelist_symbols_file_name,
                              filelist_validation_file_name, get_ds_dir,
                              get_filelist_dir, get_inference_dir,
                              get_validation_dir, inference_config_file,
                              log_inference_config, log_input_file,
                              log_map_file, log_train_config, log_train_map,
                              train_config_file, get_checkpoint_dir)
from src.waveglow.prepare_ds import load_filepaths
from src.waveglow.inference import infer
from src.common.utils import get_last_checkpoint

def init_validate_parser(parser):
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--training_dir', type=str, required=True)
  parser.add_argument('--utterance', type=str, help="Utterance name or random-val or random-val-B12", required=True)
  parser.add_argument('--hparams', type=str)
  parser.add_argument("--denoiser_strength", default=0.0, type=float, help='Removes model bias. Start with 0.1 and adjust')
  parser.add_argument("--sigma", default=1.0, type=float)
  parser.add_argument('--custom_checkpoint', type=str)
  return __main

def __main(base_dir, training_dir, utterance, hparams, denoiser_strength, sigma, custom_checkpoint):
  training_dir_path = os.path.join(base_dir, training_dir)

  if custom_checkpoint:
    checkpoint = custom_checkpoint
  else:
    checkpoint_dir = get_checkpoint_dir(training_dir_path)
    checkpoint = get_last_checkpoint(checkpoint_dir)

  infer_utterance_name = utterance

  if "random-val" in infer_utterance_name:
    tmp = infer_utterance_name.split('-')
    speaker_is_given = len(tmp) == 3
    if speaker_is_given:
      speaker = tmp[2]

    valset_path = os.path.join(get_filelist_dir(training_dir_path), filelist_validation_file_name)
    all_names = list(load_filepaths(valset_path))
    infer_utterance_name = random.choice(all_names)[0]
    while speaker_is_given:
      if speaker in infer_utterance_name:
        break
      infer_utterance_name = random.choice(all_names)
    print("Selected random validationset utterance: {}".format(infer_utterance_name))
  else:
    valset_path = os.path.join(get_filelist_dir(training_dir_path), filelist_validation_file_name)
    all_names = list(load_filepaths(valset_path))
    for x in all_names:
      if infer_utterance_name in x[0]:
        infer_utterance_name = x[0]
  basename = os.path.basename(infer_utterance_name)[:-4]

  infer_dir_path = get_validation_dir(training_dir_path, basename, checkpoint, "{}_{}".format(sigma, denoiser_strength))

  infer(training_dir_path, infer_dir_path, hparams=hparams, checkpoint=checkpoint, infer_wav_path=infer_utterance_name, denoiser_strength=denoiser_strength, sigma=sigma)

if __name__ == "__main__":
  __main(
    base_dir = '/datasets/models/taco2pt_v2',
    training_dir = 'wg_debug',
    utterance = "LJ001-0001",
    hparams = None,
    denoiser_strength = 0,
    sigma = 1,
    custom_checkpoint = None
    #utterance = "random-val",
  )