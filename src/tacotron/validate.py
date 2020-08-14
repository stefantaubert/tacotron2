import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from shutil import copyfile

import matplotlib
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize
from scipy.io import wavfile
from scipy.io.wavfile import write
from tqdm import tqdm

import torch
from src.core.common import float_to_wav, is_overamp, mel_to_numpy
from src.core.common import reset_log
from src.core.common import (parse_ds_speaker, parse_ds_speakers, parse_json,
                              stack_images_vertically)
from src.pre.mel_parser import MelParser, plot_melspec
from src.tacotron.hparams import create_hparams
from src.tacotron.model import Tacotron2
from src.tacotron.synthesizer import Synthesizer as TacoSynthesizer
from src.tacotron.train import load_model
from src.tacotron.train_io import get_checkpoint
from src.text.symbol_converter import deserialize_symbol_ids, load_from_file
from src.waveglow.inference import Synthesizer as WGSynthesizer
from src.waveglow.synthesizer import Synthesizer as WGSynthesizer
from src.tacotron.prepare_ds_ms_io import parse_all_speakers, parse_all_symbols, get_random_val_utterance, get_random_test_utterance, get_values_entry, PreparedData, PreparedDataList
from src.tacotron.validate_io import get_validation_dir, write_input_file


def init_validate_parser(parser):
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--training_dir', type=str, required=True)
  parser.add_argument('--utterance', type=str, help="Utterance id or random-val or random-val-B12", required=True)
  parser.add_argument('--hparams', type=str)
  parser.add_argument('--waveglow', type=str, required=True)
  parser.add_argument('--custom_checkpoint', type=str)
  parser.add_argument("--denoiser_strength", default=0.01, type=float, help='Removes model bias.')
  parser.add_argument("--sigma", default=0.666, type=float)
  return validate

def get_utterance(training_dir_path, utterance: str) -> PreparedData:
  if "random" in utterance:
    tmp = utterance.split('-')
    custom_speaker = tmp[2] if len(tmp) == 3 else ''
    if "test" in utterance:
      dataset_values_entry = get_random_test_utterance(training_dir_path, custom_speaker)
    elif "val" in utterance:
      dataset_values_entry = get_random_val_utterance(training_dir_path, custom_speaker)
  else:
    dataset_values_entry = get_values_entry(training_dir_path, int(utterance))
  return dataset_values_entry

def validate(base_dir, training_dir, utterance: str, hparams, waveglow, custom_checkpoint, denoiser_strength, sigma):
  training_dir_path = os.path.join(base_dir, training_dir)

  assert os.path.isfile(waveglow)


  dataset_values_entry = get_utterance(training_dir_path, utterance)
  print("Selected utterance: {} ({})".format(dataset_values_entry.i, dataset_values_entry.basename))
  
  print("Speaker is: {} ({})".format(dataset_values_entry.speaker_name, dataset_values_entry.speaker_id))

  checkpoint, checkpoint_path = get_checkpoint(training_dir_path, custom_checkpoint)
  infer_dir_path = get_validation_dir(training_dir_path, dataset_values_entry.i, dataset_values_entry.basename, checkpoint, dataset_values_entry.speaker_name)

  conv = parse_all_symbols(training_dir_path)
  print('Loaded {} symbols'.format(conv.get_symbol_ids_count()))

  n_speakers = len(parse_all_speakers(training_dir_path))
  print('Loaded {} speaker(s)'.format(n_speakers))

  print("Using tacotron model:", checkpoint_path)
  taco_synt = TacoSynthesizer(checkpoint_path=checkpoint_path, n_speakers=n_speakers, n_symbols=conv.get_symbol_ids_count(), custom_hparams=hparams)

  print("Using waveglow model:", waveglow)
  wg_synt = WGSynthesizer(checkpoint_path=waveglow, custom_hparams=None)
 
  symbol_ids = deserialize_symbol_ids(dataset_values_entry.serialized_updated_ids)
  orig_text = conv.ids_to_text(symbol_ids)
  
  write_input_file(infer_dir_path, orig_text)

  print("Inferring {}...".format(dataset_values_entry.basename))
  print("{} ({})".format(orig_text, len(symbol_ids)))
  mel_outputs, mel_outputs_postnet, alignments = taco_synt.infer(
    symbol_ids=symbol_ids,
    speaker_id=dataset_values_entry.speaker_id
  )
  synthesized_sentence = wg_synt.infer_mel(
    mel=mel_outputs_postnet,
    sigma=sigma,
    denoiser_strength=denoiser_strength
  )

  print("Saving...")

  last_dir_name = Path(infer_dir_path).parts[-1]
  output_name = "{}".format(last_dir_name)
  out_path_template = os.path.join(infer_dir_path, output_name)
  
  path_alignments_plot = "{}_alignments.png".format(out_path_template)
  path_pre_postnet_plot = "{}_pre_postnet.png".format(out_path_template)
  path_original_wav = "{}_orig.wav".format(out_path_template)
  path_original_plot = "{}_orig.png".format(out_path_template)
  path_inferred_wav = "{}_inferred.wav".format(out_path_template)
  path_inferred_plot = "{}_inferred.png".format(out_path_template)
  path_compared_plot = "{}_comparison.png".format(out_path_template)

  if is_overamp(synthesized_sentence):
    print("Overamplified output!.")
  #assert not is_overamp(synthesized_sentence)
  float_to_wav(
    wav=synthesized_sentence,
    path=path_inferred_wav,
    dtype=np.int16,
    normalize=True,
    sample_rate=taco_synt.hparams.sampling_rate
  )

  print("Finished. Saved to:", path_inferred_wav)

  print("Plotting...")

  mel_parser = MelParser(taco_synt.hparams)
  mel_orig = mel_parser.get_mel_tensor_from_file(dataset_values_entry.wav_path)

  plot_melspec(mel_to_numpy(mel_outputs_postnet), title="Inferred")
  plt.savefig(path_inferred_plot, bbox_inches='tight')

  plot_melspec(mel_orig, title="Original")
  plt.savefig(path_original_plot, bbox_inches='tight')

  plot_melspec(mel_to_numpy(mel_outputs), title="Pre-Postnet")
  plt.savefig(path_pre_postnet_plot, bbox_inches='tight')
  
  plot_melspec(mel_to_numpy(alignments).T, title="Alignments")
  plt.savefig(path_alignments_plot, bbox_inches='tight')

  stack_images_vertically([path_original_plot, path_inferred_plot, path_pre_postnet_plot, path_alignments_plot], path_compared_plot)
  copyfile(dataset_values_entry.wav_path, path_original_wav)

  print("Finished.")


if __name__ == "__main__":
  validate(
    base_dir = '/datasets/models/taco2pt_v2',
    training_dir = 'debug',
    #utterance = "1069",
    utterance = "random-val",
    waveglow = "/datasets/models/pretrained/waveglow_256channels_universal_v5.pt",
    custom_checkpoint = None,
    hparams = None,
    denoiser_strength=0.01,
    sigma=0.666
  )
