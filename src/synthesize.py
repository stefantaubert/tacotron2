import argparse
import os
import sys
import time
from pathlib import Path
from shutil import copyfile

import matplotlib
import matplotlib.pylab as plt
import numpy as np
from nltk.tokenize import sent_tokenize
from scipy.io import wavfile
from scipy.io.wavfile import write
from tqdm import tqdm

import torch
from src.common.audio.utils import float_to_wav, mel_to_numpy
from src.common.utils import parse_ds_speakers, parse_json
from src.paths import (filelist_speakers_name, get_checkpoint_dir,
                       get_filelist_dir, get_symbols_path,
                       inference_input_file_name,
                       inference_input_symbols_file_name)
from src.tacotron.hparams import create_hparams
from src.tacotron.model import Tacotron2
from src.tacotron.plot_mel import plot_melspec, stack_images_vertically
from src.tacotron.synthesizer import Synthesizer as TacoSynthesizer
from src.common.utils import get_last_checkpoint
from src.tacotron.train import load_model
from src.text.symbol_converter import deserialize_symbol_ids, load_from_file
from src.waveglow.inference import Synthesizer as WGSynthesizer
from src.waveglow.mel2samp import MelParser
from src.waveglow.synthesizer import Synthesizer as WGSynthesizer

matplotlib.use("Agg")

def validate(training_dir_path: str, infer_dir_path: str, hparams: str, waveglow: str, checkpoint_path: str, infer_data: tuple) -> None:
  n_symbols, conv = get_symbols_count(training_dir_path)
  print('Loaded {} symbols'.format(n_symbols))

  n_speakers, _ = get_speakers_count(training_dir_path)
  print('Loaded {} speaker(s)'.format(n_speakers))

  print("Using tacotron model:", checkpoint_path)
  taco_synt = TacoSynthesizer(checkpoint_path=checkpoint_path, n_speakers=n_speakers, n_symbols=n_symbols, custom_hparams=hparams)

  print("Using waveglow model:", waveglow)
  wg_synt = WGSynthesizer(checkpoint_path=waveglow, custom_hparams=None)
 
  utt_name, serialized_symbol_ids, wav_orig_path, final_speaker_id = infer_data

  symbol_ids = deserialize_symbol_ids(serialized_symbol_ids)
  orig_text = conv.ids_to_text(symbol_ids)
  
  with open(os.path.join(infer_dir_path, inference_input_file_name), 'w', encoding='utf-8') as f:
    f.writelines([orig_text])
  
  print("Inferring {}...".format(utt_name))
  print("{} ({})".format(orig_text, len(symbol_ids)))
  mel_outputs, mel_outputs_postnet, alignments = taco_synt.infer(
    symbol_ids=symbol_ids,
    speaker_id=final_speaker_id
  )
  synthesized_sentence = wg_synt.infer_mel(
    mel=mel_outputs_postnet,
    sigma=0.666,
    denoiser_strength=10**-2
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
  mel_orig = mel_parser.get_mel(wav_orig_path)[0]

  plot_melspec(mel_to_numpy(mel_outputs_postnet), title="Inferred")
  plt.savefig(path_inferred_plot, bbox_inches='tight')

  plot_melspec(mel_orig, title="Original")
  plt.savefig(path_original_plot, bbox_inches='tight')

  plot_melspec(mel_to_numpy(mel_outputs), title="Pre-Postnet")
  plt.savefig(path_pre_postnet_plot, bbox_inches='tight')
  
  plot_melspec(mel_to_numpy(alignments).T, title="Alignments")
  plt.savefig(path_alignments_plot, bbox_inches='tight')

  stack_images_vertically([path_original_plot, path_inferred_plot, path_pre_postnet_plot, path_alignments_plot], path_compared_plot)
  copyfile(wav_orig_path, path_original_wav)

  print("Finished.")

def get_speakers_count(training_dir_path: str) -> int:
  speakers_file = os.path.join(get_filelist_dir(training_dir_path), filelist_speakers_name)
  all_speakers = parse_json(speakers_file)
  n_speakers = len(all_speakers)
  return n_speakers, all_speakers

def get_symbols_count(training_dir_path: str):
  conv = load_from_file(get_symbols_path(training_dir_path))
  n_symbols = conv.get_symbol_ids_count()
  return n_symbols, conv

def get_speaker_id(all_speakers: list, speaker_name):
  final_speaker_id = -1
  for ds_speaker, speaker_id in all_speakers.items():
    if ds_speaker == speaker_name:
      final_speaker_id = speaker_id
      break
    
  if final_speaker_id == -1:
    raise Exception("Speaker {} not available!".format(speaker_name))

  return final_speaker_id

def infer(training_dir_path: str, infer_dir_path: str, hparams, waveglow: str, checkpoint_path: str, speaker: str, analysis: bool, sentence_pause_s: float, sigma: float, denoiser_strength: float, sampling_rate: int):
  # Speed is: 1min inference for 3min wav result
  n_symbols, conv = get_symbols_count(training_dir_path)
  print('Loaded {} symbols'.format(n_symbols))

  n_speakers, all_speakers = get_speakers_count(training_dir_path)
  print('Loaded {} speaker(s)'.format(n_speakers))

  print("Using tacotron model:", checkpoint_path)
  taco_synt = TacoSynthesizer(checkpoint_path=checkpoint_path, n_speakers=n_speakers, n_symbols=n_symbols, custom_hparams=hparams)

  print("Using waveglow model:", waveglow)
  wg_synt = WGSynthesizer(checkpoint_path=waveglow, custom_hparams=None)
 
  with open(os.path.join(infer_dir_path, inference_input_symbols_file_name), 'r', encoding='utf-8') as f:
    serialized_symbol_ids_sentences = f.readlines()

  sentence_pause_samples_count = int(round(sampling_rate * sentence_pause_s, 0))
  sentence_pause_samples = np.zeros(shape=sentence_pause_samples_count)

  print("Inferring...")

  output = np.array([])

  final_speaker_id = get_speaker_id(all_speakers, speaker)

  last_dir_name = Path(infer_dir_path).parts[-1]

  mel_plot_files = []
  alignment_plots = []
  pre_post_plots = []

  for i, serialized_symbol_ids in tqdm(enumerate(serialized_symbol_ids_sentences), total=len(serialized_symbol_ids_sentences)):
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
  mel = mel_parser.get_mel(out_path)[0]
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
