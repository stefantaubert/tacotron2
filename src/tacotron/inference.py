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
from src.common.audio.utils import float_to_wav, is_overamp, mel_to_numpy
from src.common.train_log import reset_log
from src.common.utils import (get_last_checkpoint,
                              parse_ds_speaker, parse_ds_speakers, parse_json,
                              stack_images_vertically)
from src.pre.mel_parser import MelParser, plot_melspec
from src.tacotron.hparams import create_hparams
from src.tacotron.model import Tacotron2
from src.tacotron.synthesizer import Synthesizer as TacoSynthesizer
from src.tacotron.train import load_model
from src.tacotron.txt_pre import process_input_text
from src.text.symbol_converter import deserialize_symbol_ids, load_from_file
from src.waveglow.inference import Synthesizer as WGSynthesizer
from src.waveglow.synthesizer import Synthesizer as WGSynthesizer
from src.tacotron.prepare_ds_ms_io import parse_all_speakers, parse_all_symbols, get_speaker_id_from_name

def infer(training_dir_path: str, infer_dir_path: str, hparams, waveglow: str, checkpoint_path: str, speaker: str, analysis: bool, sentence_pause_s: float, sigma: float, denoiser_strength: float, sampling_rate: int):
  # Speed is: 1min inference for 3min wav result
  conv = parse_all_symbols(training_dir_path)
  print('Loaded {} symbols'.format(conv.get_symbol_ids_count()))

  n_speakers = len(parse_all_speakers(training_dir_path))
  print('Loaded {} speaker(s)'.format(n_speakers))

  print("Using tacotron model:", checkpoint_path)
  taco_synt = TacoSynthesizer(checkpoint_path=checkpoint_path, n_speakers=n_speakers, n_symbols=conv.get_symbol_ids_count(), custom_hparams=hparams)

  print("Using waveglow model:", waveglow)
  wg_synt = WGSynthesizer(checkpoint_path=waveglow, custom_hparams=None)
 
  with open(os.path.join(infer_dir_path, inference_input_symbols_file_name), 'r', encoding='utf-8') as f:
    serialized_symbol_ids_sentences = f.readlines()

  sentence_pause_samples_count = int(round(sampling_rate * sentence_pause_s, 0))
  sentence_pause_samples = np.zeros(shape=sentence_pause_samples_count)

  print("Inferring...")

  output = np.array([])

  final_speaker_id = get_speaker_id_from_name(training_dir_path, speaker)

  last_dir_name = Path(infer_dir_path).parts[-1]

  mel_plot_files = []
  alignment_plots = []
  pre_post_plots = []

  for i, serialized_symbol_ids in enumerate(tqdm(serialized_symbol_ids_sentences)):
    #print(sentence_symbols)
    symbol_ids = deserialize_symbol_ids(serialized_symbol_ids)
    print("{} ({})".format(conv.ids_to_text(symbol_ids), len(symbol_ids)))
    mel_outputs, mel_outputs_postnet, alignments = taco_synt.infer(
      symbol_ids=symbol_ids,
      speaker_id=final_speaker_id
    )
    synthesized_sentence = wg_synt.infer_mel(
      mel=mel_outputs_postnet,
      sigma=sigma,
      denoiser_strength=denoiser_strength
    )

    if analysis:
      path_inferred_wav = os.path.join(infer_dir_path, "{}.wav".format(i))
      path_inferred_plot = os.path.join(infer_dir_path, "{}.png".format(i))
      path_pre_postnet_plot = os.path.join(infer_dir_path, "{}_pre_post.png".format(i))
      path_alignments_plot = os.path.join(infer_dir_path, "{}_alignments.png".format(i))
       
      float_to_wav(
        wav=synthesized_sentence,
        path=path_inferred_wav,
        dtype=np.int16,
        normalize=True,
        sample_rate=sampling_rate
      )

      plot_melspec(mel_to_numpy(mel_outputs_postnet), title="{}: {}".format(last_dir_name, i))
      plt.savefig(path_inferred_plot, bbox_inches='tight')

      plot_melspec(mel_to_numpy(mel_outputs), title="{}: Pre-Postnet {}".format(last_dir_name, i))
      plt.savefig(path_pre_postnet_plot, bbox_inches='tight')
      
      plot_melspec(mel_to_numpy(alignments).T, title="{}: Alignments {}".format(last_dir_name, i))
      plt.savefig(path_alignments_plot, bbox_inches='tight')

      mel_plot_files.append(path_inferred_plot)
      pre_post_plots.append(path_pre_postnet_plot)
      alignment_plots.append(path_alignments_plot)

    output = np.concatenate((output, synthesized_sentence, sentence_pause_samples), axis=0)

  print("Saving...")
  output_name = "{}.wav".format(last_dir_name)
  out_path = os.path.join(infer_dir_path, output_name)

  if is_overamp(output):
    print("Overamplified output!.")
  float_to_wav(
    wav=output,
    path=out_path,
    dtype=np.int16,
    normalize=True,
    sample_rate=sampling_rate
  )
  print("Finished. Saved to:", out_path)

  print("Plotting...")

  mel_parser = MelParser(taco_synt.hparams)
  mel = mel_parser.get_mel_tensor_from_file(out_path)
  plot_melspec(mel, title=last_dir_name)
  output_name = "{}_h.png".format(last_dir_name)
  out_path = os.path.join(infer_dir_path, output_name)
  plt.savefig(out_path, bbox_inches='tight')

  if analysis:
    out_path = os.path.join(infer_dir_path, "{}_v.png".format(last_dir_name))
    stack_images_vertically(mel_plot_files, out_path)

    out_path = os.path.join(infer_dir_path, "{}_v_pre_post.png".format(last_dir_name))
    stack_images_vertically(pre_post_plots, out_path)

    out_path = os.path.join(infer_dir_path, "{}_v_alignments.png".format(last_dir_name))
    stack_images_vertically(alignment_plots, out_path)

def init_inference_parser(parser):
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--training_dir', type=str, required=True)
  parser.add_argument('--ipa', action='store_true')
  parser.add_argument('--text', type=str, required=True)
  parser.add_argument('--lang', type=str, choices=["ipa", "en", "chn", "ger"], required=True)
  parser.add_argument('--ignore_tones', action='store_true')
  parser.add_argument('--ignore_arcs', action='store_true')
  parser.add_argument('--weights_map', type=str)
  parser.add_argument('--speaker', type=str, required=True)
  #parser.add_argument('--subset_id', type=str)
  parser.add_argument('--hparams', type=str)
  parser.add_argument('--waveglow', type=str, required=True)
  parser.add_argument('--custom_checkpoint', type=str)
  parser.add_argument('--sentence_pause_s', type=float, default=0.5)
  parser.add_argument('--sigma', type=float, default=0.666)
  parser.add_argument('--denoiser_strength', type=float, default=0.01)
  parser.add_argument('--sampling_rate', type=float, default=22050)
  parser.add_argument('--analysis', action='store_true')
  return __main

def __main(base_dir: str, training_dir: str, ipa: bool, text: str, lang: str, ignore_tones: bool, ignore_arcs:bool, weights_map: str, speaker: str, hparams: str, waveglow: str, custom_checkpoint: str, sentence_pause_s: float, sigma: float, denoiser_strength: float, sampling_rate: float, analysis: bool):
  training_dir_path = os.path.join(base_dir, training_dir)

  assert os.path.isfile(text)
  assert os.path.isfile(waveglow)

  print("Infering text from:", text)
  input_name = os.path.splitext(os.path.basename(text))[0]
  checkpoint_dir = get_checkpoint_dir(training_dir_path)
  if custom_checkpoint:
    checkpoint = custom_checkpoint
  else:
    checkpoint = get_last_checkpoint(checkpoint_dir)
  checkpoint_path = os.path.join(get_checkpoint_dir(training_dir_path), checkpoint)
  
  speaker_name = parse_ds_speaker(speaker)[1]
  infer_dir_path = get_inference_dir(training_dir_path, input_name, checkpoint, speaker_name)
  # TODO logging
  #log_inference_config(infer_dir_path, args)
  log_input_file(infer_dir_path, text)

  if weights_map:
    assert os.path.isfile(weights_map)
    print("Using mapping from:", weights_map)
    log_map_file(infer_dir_path, weights_map)
  else:
    print("Using no mapping.")

  process_input_text(
    training_dir_path,
    infer_dir_path,
    ipa=ipa,
    ignore_tones=ignore_tones,
    ignore_arcs=ignore_arcs,
    subset_id=0,
    lang=lang,
    use_map=bool(weights_map)
  )

  infer(
    training_dir_path=training_dir_path,
    infer_dir_path=infer_dir_path,
    hparams=hparams,
    waveglow=waveglow,
    checkpoint_path=checkpoint_path,
    speaker=speaker,
    analysis=analysis,
    sentence_pause_s=sentence_pause_s,
    sigma=sigma,
    denoiser_strength=denoiser_strength,
    sampling_rate=sampling_rate
  )


if __name__ == "__main__":
  __main(
    base_dir = '/datasets/models/taco2pt_v2',
    #training_dir = 'ljs_ipa_ms_from_scratch',
    training_dir = 'debug',
    ipa = True,
    text = "examples/chn/north.txt",
    lang = "chn",
    #text = "examples/ger/nord.txt",
    #lang = "ger",
    #text = "examples/ipa/north_sven_orig.txt",
    #lang = "ipa",
    #text = "examples/en/ljs_0001.txt",
    #lang = "en",
    weights_map = "maps/inference/chn_v1.json",
    #weights_map = "maps/inference/en_v1.json",
    ignore_tones = False,
    ignore_arcs = True,
    #speakers = 'thchs_v5,B2;thchs_v5,A2',
    #speaker = 'ljs_ipa_v2,1',
    speaker = 'thchs_mel_v1,D31',
    #waveglow = "/datasets/models/pretrained/waveglow_256channels_universal_v5.pt",
    waveglow = "/datasets/models/pretrained/waveglow_256channels_universal_v5_out.pt",
    analysis = True,
    #denoiser_strength = 0.5,
    sigma = 0.666,
    hparams = None,
    custom_checkpoint = None,
    sentence_pause_s = 0.5,
    denoiser_strength = 0,
    sampling_rate = 22050,

  )
