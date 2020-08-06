import argparse
import json
import os
import random
from shutil import copyfile

import pandas as pd

from src.common.train_log import reset_log
from src.common.utils import get_last_checkpoint, parse_ds_speaker
from src.waveglow.inference import infer
from src.waveglow.prepare_ds_io import (PreparedData, PreparedDataList,
                                        get_random_test_utterance,
                                        get_random_val_utterance,
                                        get_values_entry)


def init_validate_parser(parser):
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--training_dir', type=str, required=True)
  parser.add_argument('--utterance', type=str, help="Utterance name or random-val or random-val-B12", required=True)
  parser.add_argument('--hparams', type=str)
  parser.add_argument("--denoiser_strength", default=0.0, type=float, help='Removes model bias. Start with 0.1 and adjust')
  parser.add_argument("--sigma", default=1.0, type=float)
  parser.add_argument('--custom_checkpoint', type=str)
  return __main

def get_utterance(training_dir_path, utterance: str) -> PreparedData:
  if "random" in utterance:
    tmp = utterance.split('-')
    custom_speaker = tmp[2] if len(tmp) == 3 else None
    if "test" in utterance:
      dataset_values_entry = get_random_test_utterance(training_dir_path, custom_speaker)
    elif "val" in utterance:
      dataset_values_entry = get_random_val_utterance(training_dir_path, custom_speaker)
  else:
    dataset_values_entry = get_values_entry(int(utterance))
  return dataset_values_entry

def __main(base_dir, training_dir, utterance, hparams, denoiser_strength, sigma, custom_checkpoint):
  training_dir_path = os.path.join(base_dir, training_dir)

  if custom_checkpoint:
    checkpoint = custom_checkpoint
  else:
    checkpoint_dir = get_checkpoint_dir(training_dir_path)
    checkpoint = get_last_checkpoint(checkpoint_dir)

  dataset_values_entry = get_utterance(training_dir_path, utterance)

  infer_dir_path = get_validation_dir(training_dir_path, dataset_values_entry.i, dataset_values_entry.basename, checkpoint, "{}_{}".format(sigma, denoiser_strength))

  infer(training_dir_path, infer_dir_path, hparams=hparams, checkpoint=checkpoint, infer_wav_path=dataset_values_entry.wav_path, denoiser_strength=denoiser_strength, sigma=sigma)

if __name__ == "__main__":
  __main(
    base_dir = '/datasets/models/taco2pt_v2',
    training_dir = 'wg_debug',
    utterance = "random-val",
    hparams = None,
    denoiser_strength = 0,
    sigma = 1,
    custom_checkpoint = None
    #utterance = "random-val",
  )
