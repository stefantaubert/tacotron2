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
from src.common.audio.utils import float_to_wav

import torch
from src.common.utils import parse_ds_speakers, parse_json
from src.script_paths import (filelist_speakers_name, get_checkpoint_dir,
                              get_filelist_dir, get_symbols_path,
                              inference_input_file_name,
                              inference_input_symbols_file_name)
from src.tacotron.hparams import create_hparams
from src.tacotron.model import Tacotron2
from src.tacotron.script_plot_mel import (plot_melspec, stack_images_vertically)
from src.tacotron.train import get_last_checkpoint, load_model
from src.text.symbol_converter import deserialize_symbol_ids, load_from_file
from src.waveglow.inference import Synthesizer as WGSynthesizer
from src.waveglow.mel2samp import MelParser

matplotlib.use("Agg")

# TODO refactor in own tacotron synthesizer class
class Synthesizer():
  def __init__(self, hparams, checkpoint_path, waveglow_path):
    super().__init__()
    self.hparams = hparams
    # Load model from checkpoint
    
    self.model = load_model(self.hparams)
    torch_model = torch.load(checkpoint_path)
    state_dict = torch_model['state_dict']
    self.model.load_state_dict(state_dict)
    self.model.cuda().eval().half()

    self.synthesizer = WGSynthesizer()
    self.synthesizer.load_model(waveglow_path, for_taco_infer=True)

  def infer(self, symbols, speaker_id: int, sigma, denoiser_strength):
    sequence = np.array([symbols])
    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
    speaker_id = torch.IntTensor([speaker_id]).cuda().long()

    mel_outputs, mel_outputs_postnet, _, alignments = self.model.inference(sequence, speaker_id)
    # plot_data((mel_outputs.float().data.cpu().numpy()[0], mel_outputs_postnet.float().data.cpu().numpy()[0], alignments.float().data.cpu().numpy()[0].T))
    audio = self.synthesizer.infer_mel(mel_outputs_postnet, sigma, denoiser_strength)
   
    return audio

def validate(training_dir_path: str, infer_dir_path: str, hparams, waveglow: str, checkpoint: str, infer_data: tuple) -> None:
  hparams = create_hparams(hparams)

  conv = load_from_file(get_symbols_path(training_dir_path))
  n_symbols = conv.get_symbol_ids_count()

  speakers_file = os.path.join(get_filelist_dir(training_dir_path), filelist_speakers_name)
  all_speakers = parse_json(speakers_file)
  n_speakers = len(all_speakers)

  print('Loaded {} symbols'.format(n_symbols))
  print('Loaded {} speaker(s)'.format(n_speakers))

  hparams.n_symbols = n_symbols
  hparams.n_speakers = n_speakers

  checkpoint_path = os.path.join(get_checkpoint_dir(training_dir_path), checkpoint)
  print("Using model:", checkpoint_path)
  synt = Synthesizer(hparams, checkpoint_path, waveglow)
  mel_parser = MelParser(hparams)

  utt_name, serialized_symbol_ids, wav_orig_path, final_speaker_id = infer_data
  print("Inferring {}...".format(utt_name))
  symbol_ids = deserialize_symbol_ids(serialized_symbol_ids)
  orig_text = conv.ids_to_text(symbol_ids)
  print("{} ({})".format(orig_text, len(symbol_ids)))

  with open(os.path.join(infer_dir_path, inference_input_file_name), 'w', encoding='utf-8') as f:
    f.writelines([orig_text])

  synthesized_sentence = synt.infer(symbol_ids, final_speaker_id, sigma=0.666, denoiser_strength=10**-2)

  print("Saving...")
  last_dir_name = Path(infer_dir_path).parts[-1]
  output_name = "{}".format(last_dir_name)
  out_path_template = os.path.join(infer_dir_path, output_name)
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
    sample_rate=hparams.sampling_rate
  )

  print("Finished. Saved to:", path_inferred_wav)
  print("Plotting...")
  mel_inferred, _ = mel_parser.get_mel(path_inferred_wav)
  mel_orig, _ = mel_parser.get_mel(wav_orig_path)

  plot_melspec(mel_inferred, title="Inferred")
  plt.savefig(path_inferred_plot, bbox_inches='tight')
  plot_melspec(mel_orig, title="Original")
  plt.savefig(path_original_plot, bbox_inches='tight')

  stack_images_vertically([path_original_plot, path_inferred_plot], path_compared_plot)

  # plot_melspec([mel_orig, mel_inferred], titles=["Original", "Inferred"])
  # plt.savefig(path_compared_plot, bbox_inches='tight')
  copyfile(wav_orig_path, path_original_wav)
  print("Finished.")

def infer(training_dir_path: str, infer_dir_path: str, hparams, waveglow: str, checkpoint: str, speaker: str, analysis: bool, sentence_pause_s: float, sigma: float, denoiser_strength: float):
  hparams = create_hparams(hparams)

  conv = load_from_file(get_symbols_path(training_dir_path))
  n_symbols = conv.get_symbol_ids_count()

  speakers_file = os.path.join(get_filelist_dir(training_dir_path), filelist_speakers_name)
  all_speakers = parse_json(speakers_file)
  n_speakers = len(all_speakers)

  print('Loaded {} symbols'.format(n_symbols))
  print('Loaded {} speaker(s)'.format(n_speakers))

  hparams.n_symbols = n_symbols
  hparams.n_speakers = n_speakers

  with open(os.path.join(infer_dir_path, inference_input_symbols_file_name), 'r', encoding='utf-8') as f:
    serialized_symbol_ids_sentences = f.readlines()

  hparams.n_symbols = n_symbols
  hparams.n_speakers = n_speakers

  checkpoint_path = os.path.join(get_checkpoint_dir(training_dir_path), checkpoint)
  print("Using model:", checkpoint_path)
  synt = Synthesizer(hparams, checkpoint_path, waveglow)
  mel_parser = MelParser(hparams)

  #complete_text = [item for sublist in sentences_symbols for item in sublist]
  #print(complete_text)
  #res = synt.infer(complete_text, "aei")
  #to_wav("out/complete_x.wav", res, synt.hparams.sampling_rate)
  #print("exit")

  # Speed is: 1min inference for 3min wav result

  sentence_pause_samples_count = int(round(hparams.sampling_rate * sentence_pause_s, 0))
  sentence_pause_samples = np.zeros(shape=sentence_pause_samples_count)

  print("Inferring...")

  output = np.array([])

  final_speaker_id = -1
  for ds_speaker, speaker_id in all_speakers.items():
    if ds_speaker == speaker:
      final_speaker_id = speaker_id
      break
    
  if final_speaker_id == -1:
    raise Exception("Speaker {} not available!".format(speaker))

  last_dir_name = Path(infer_dir_path).parts[-1]

  mel_plot_files = []

  for i, serialized_symbol_ids in tqdm(enumerate(serialized_symbol_ids_sentences), total=len(serialized_symbol_ids_sentences)):
    #print(sentence_symbols)
    symbol_ids = deserialize_symbol_ids(serialized_symbol_ids)
    print("{} ({})".format(conv.ids_to_text(symbol_ids), len(symbol_ids)))
    synthesized_sentence = synt.infer(
      symbols=symbol_ids,
      speaker_id=final_speaker_id,
      sigma=sigma,
      denoiser_strength=denoiser_strength
    )
    if analysis:
      out_path = os.path.join(infer_dir_path, "{}.wav".format(i))
      float_to_wav(
        wav=synthesized_sentence,
        path=out_path,
        dtype=np.int16,
        normalize=True,
        sample_rate=hparams.sampling_rate
      )
      mel, _ = mel_parser.get_mel(out_path)
      plot_melspec(mel, title="{}: {}".format(last_dir_name, i))
      out_path = os.path.join(infer_dir_path, "{}.png".format(i))
      plt.savefig(out_path, bbox_inches='tight')
      mel_plot_files.append(out_path)
    output = np.concatenate((output, synthesized_sentence, sentence_pause_samples), axis=0)
    #print(output)

  if analysis:
    out_path = os.path.join(infer_dir_path, "{}_v.png".format(last_dir_name))
    stack_images_vertically(mel_plot_files, out_path)

  print("Saving...")
  output_name = "{}.wav".format(last_dir_name)
  out_path = os.path.join(infer_dir_path, output_name)
  float_to_wav(
    wav=output,
    path=out_path,
    dtype=np.int16,
    normalize=True,
    sample_rate=hparams.sampling_rate
  )
  print("Finished. Saved to:", out_path)
  print("Plotting...")
  mel, _ = mel_parser.get_mel(out_path)
  plot_melspec(mel, title=last_dir_name)
  output_name = "{}_h.png".format(last_dir_name)
  out_path = os.path.join(infer_dir_path, output_name)
  plt.savefig(out_path, bbox_inches='tight')
  plt.show()
