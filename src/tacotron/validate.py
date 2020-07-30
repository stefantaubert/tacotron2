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
                              filelist_symbols_file_name, get_checkpoint_dir,
                              filelist_validation_file_name, filelist_test_file_name, get_ds_dir,
                              get_filelist_dir, get_inference_dir,
                              get_validation_dir, inference_config_file,
                              log_inference_config, log_input_file,
                              log_map_file, log_train_config, log_train_map,
                              train_config_file)
from src.tacotron.prepare_ds import prepare
from src.tacotron.plot_embeddings import analyse
from src.synthesize import validate
from src.tacotron.train import start_train
from src.common.utils import get_last_checkpoint
from src.tacotron.txt_pre import process_input_text


def init_validate_parser(parser):
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--training_dir', type=str, required=True)
  parser.add_argument('--utterance', type=str, help="Utterance name or random-val or random-val-B12", required=True)
  parser.add_argument('--hparams', type=str)
  parser.add_argument('--waveglow', type=str, required=True)
  parser.add_argument('--custom_checkpoint', type=str)
  return __main


def __main(base_dir, training_dir, utterance, hparams, waveglow, custom_checkpoint):
  training_dir_path = os.path.join(base_dir, training_dir)

  assert os.path.isfile(waveglow)

  if custom_checkpoint:
    checkpoint = custom_checkpoint
  else:
    checkpoint_dir = get_checkpoint_dir(training_dir_path)
    checkpoint = get_last_checkpoint(checkpoint_dir)

  preprocessed_path = os.path.join(get_filelist_dir(training_dir_path), ds_preprocessed_file_name)

  if "random-val" in utterance:
    tmp = utterance.split('-')
    speaker_is_given = len(tmp) == 3
    if speaker_is_given:
      speaker = tmp[2]

    valset_path = os.path.join(get_filelist_dir(training_dir_path), filelist_validation_file_name)
    valset = pd.read_csv(valset_path, header=None, sep=csv_separator)
    all_names = list(get_utterance_names_csv(valset))
    infer_utterance_name = random.choice(all_names)
    while speaker_is_given:
      if speaker in infer_utterance_name:
        break
      infer_utterance_name = random.choice(all_names)
    print("Selected random validationset utterance: {}".format(infer_utterance_name))
  elif "random-test" in utterance:
    tmp = utterance.split('-')
    speaker_is_given = len(tmp) == 3
    if speaker_is_given:
      speaker = tmp[2]

    testset_path = os.path.join(get_filelist_dir(training_dir_path), filelist_test_file_name)
    testset = pd.read_csv(testset_path, header=None, sep=csv_separator)
    all_names = list(get_utterance_names_csv(testset))
    infer_utterance_name = random.choice(all_names)
    while speaker_is_given:
      if speaker in infer_utterance_name:
        break
      infer_utterance_name = random.choice(all_names)
    print("Selected random testset utterance: {}".format(infer_utterance_name))
  else:
    infer_utterance_name = utterance

  data = pd.read_csv(preprocessed_path, header=None, sep=csv_separator)
  infer_data = None
  for i, row in data.iterrows():
    utt_name = row[utt_name_col]
    if utt_name == infer_utterance_name:
      symbs = row[symbols_str_col]
      wav_path = row[wavpath_col]
      speaker_id = row[speaker_id_col]
      speaker_name = row[speaker_name_col]
      infer_data = (utt_name, symbs, wav_path, speaker_id)
      break

  if not infer_data:
    raise Exception("Utterance {} was not found".format(infer_utterance_name))
  
  print("Speaker is: {} ({})".format(speaker_name, str(speaker_id)))
  infer_dir_path = get_validation_dir(training_dir_path, infer_utterance_name, checkpoint, speaker_name)
  checkpoint_path = os.path.join(get_checkpoint_dir(training_dir_path), checkpoint)

  validate(training_dir_path, infer_dir_path, hparams=hparams, waveglow=waveglow, checkpoint_path=checkpoint_path, infer_data=infer_data)


if __name__ == "__main__":
  __main(
    base_dir = '/datasets/models/taco2pt_v2',
    training_dir = 'ljs_ipa_ms_from_scratch',
    utterance = "LJ002-0205",
    #utterance = "random-val",
    waveglow = "/datasets/models/pretrained/waveglow_256channels_universal_v5.pt",
    custom_checkpoint = None
  )