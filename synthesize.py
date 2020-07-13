import sys

import argparse
import matplotlib
matplotlib.use("Agg")

import matplotlib.pylab as plt
import numpy as np
from scipy.io import wavfile
from shutil import copyfile
import time

import os
import torch
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from text.symbol_converter import load_from_file, deserialize_symbol_ids
from paths import get_symbols_path, inference_input_symbols_file_name, get_checkpoint_dir, inference_input_file_name
from train import get_last_checkpoint
from utils import parse_ds_speakers
from plot_mel import Mel2Samp, plot_melspecs, get_segment, get_audio

# to load denoiser, glow etc.
sys.path.append('waveglow/')
from denoiser import Denoiser
from pathlib import Path

from hparams import create_hparams
from model import Tacotron2
from train import load_model
from scipy.io.wavfile import write

def to_wav(path, data, sr):
  wav = data
  wav_max = np.max(np.abs(wav))
  amp = (2**15 - 1) / max(10**-2, wav_max)
  # to prevent overmodulation
  wav *= amp
  #wavfile.write(path, rate=sr, data=wav.astype(np.int16))
  wav_int = wav.astype(np.int16)
  #wav_int += wav_int.min
  write(path, sr, wav_int)

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

    # Load WaveGlow for mel2audio synthesis and denoiser
    self.waveglow = torch.load(waveglow_path)['model']
    self.waveglow.cuda().eval().half()
    for k in self.waveglow.convinv:
      k.float()
    self.denoiser = Denoiser(self.waveglow)

  def infer(self, symbols, speaker_id: int):
    sequence = np.array([symbols])
    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
    speaker_id = torch.IntTensor([speaker_id]).cuda().long()

    # Decode text input and plot results
    mel_outputs, mel_outputs_postnet, _, alignments = self.model.inference(sequence, speaker_id)
    # plot_data((mel_outputs.float().data.cpu().numpy()[0], mel_outputs_postnet.float().data.cpu().numpy()[0], alignments.float().data.cpu().numpy()[0].T))

    with torch.no_grad():
      audio = self.waveglow.infer(mel_outputs_postnet, sigma=0.666)
    #res = audio[0].data.cpu().numpy()
    #print("Saving {}...".format(dest_name))
    #to_wav("/tmp/{}.wav".format(dest_name), res, self.hparams.sampling_rate)

    # (Optional) Remove WaveGlow bias
    audio_denoised = self.denoiser(audio, strength=10**-2)[:, 0]
    res = audio_denoised.cpu().numpy()[0]
    #to_wav("/tmp/{}_denoised.wav".format(dest_name), res, self.hparams.sampling_rate)
    return res

def validate(training_dir_path: str, infer_dir_path: str, hparams, waveglow: str, checkpoint: str, infer_data: tuple, speaker_count: int) -> None:
  hparams = create_hparams(hparams)

  conv = load_from_file(get_symbols_path(training_dir_path))
  n_symbols = conv.get_symbol_ids_count()
  print('Loaded {} symbols'.format(n_symbols))
  print('Loaded {} speaker(s)'.format(speaker_count))

  hparams.n_symbols = n_symbols
  hparams.n_speakers = speaker_count

  checkpoint_path = os.path.join(get_checkpoint_dir(training_dir_path), checkpoint)
  print("Using model:", checkpoint_path)
  synt = Synthesizer(hparams, checkpoint_path, waveglow)
  plotter = Mel2Samp(hparams)

  utt_name, serialized_symbol_ids, wav_orig_path, final_speaker_id = infer_data
  print("Inferring {}...".format(utt_name))
  symbol_ids = deserialize_symbol_ids(serialized_symbol_ids)
  orig_text = conv.ids_to_text(symbol_ids)
  print("{} ({})".format(orig_text, len(symbol_ids)))

  with open(os.path.join(infer_dir_path, inference_input_file_name), 'w') as f:
    f.writelines([orig_text])

  synthesized_sentence = synt.infer(symbol_ids, final_speaker_id)

  print("Saving...")
  last_dir_name = Path(infer_dir_path).parts[-1]
  output_name = "{}".format(last_dir_name)
  out_path_template = os.path.join(infer_dir_path, output_name)
  path_original_wav = "{}_orig.wav".format(out_path_template)
  path_original_plot = "{}_orig.png".format(out_path_template)
  path_inferred_wav = "{}_inferred.wav".format(out_path_template)
  path_inferred_plot = "{}_inferred.png".format(out_path_template)
  path_compared_plot = "{}_comparison.png".format(out_path_template)

  to_wav(path_inferred_wav, synthesized_sentence, hparams.sampling_rate)
  print("Finished. Saved to:", path_inferred_wav)
  print("Plotting...")
  wav_inferred = get_audio(path_inferred_wav)
  wav_orig = get_audio(wav_orig_path)
  #wav = get_segment(wav)
  #wav = get_segment(wav)
  mel_inferred = plotter.get_mel(wav_inferred)
  mel_orig = plotter.get_mel(wav_orig)

  plot_melspecs([mel_inferred])
  plt.savefig(path_inferred_plot, bbox_inches='tight')
  plot_melspecs([mel_orig])
  plt.savefig(path_original_plot, bbox_inches='tight')
  plot_melspecs([mel_orig, mel_inferred], titles=["Original", "Inferred"])
  plt.savefig(path_compared_plot, bbox_inches='tight')
  copyfile(wav_orig_path, path_original_wav)
  print("Finished.")

def infer(training_dir_path: str, infer_dir_path: str, hparams, waveglow: str, checkpoint: str, speakers: str, speaker: str):
  hparams = create_hparams(hparams)

  conv = load_from_file(get_symbols_path(training_dir_path))
  n_symbols = conv.get_symbol_ids_count()
  print('Loaded {} symbols'.format(n_symbols))
  ds_speakers = parse_ds_speakers(speakers)
  n_speakers = len(ds_speakers)
  print('Loaded {} speaker(s)'.format(n_speakers))

  with open(os.path.join(infer_dir_path, inference_input_symbols_file_name), 'r') as f:
    serialized_symbol_ids_sentences = f.readlines()

  hparams.n_symbols = n_symbols
  hparams.n_speakers = n_speakers

  checkpoint_path = os.path.join(get_checkpoint_dir(training_dir_path), checkpoint)
  print("Using model:", checkpoint_path)
  synt = Synthesizer(hparams, checkpoint_path, waveglow)
  plotter = Mel2Samp(hparams)

  #complete_text = [item for sublist in sentences_symbols for item in sublist]
  #print(complete_text)
  #res = synt.infer(complete_text, "aei")
  #to_wav("out/complete_x.wav", res, synt.hparams.sampling_rate)
  #print("exit")

  # Speed is: 1min inference for 3min wav result

  sentence_pause_sec = 0.5
  sentence_pause_samples_count = int(round(hparams.sampling_rate * sentence_pause_sec, 0))
  sentence_pause_samples = np.zeros(shape=sentence_pause_samples_count)

  print("Inferring...")

  output = np.array([])

  final_speaker_id = -1
  dest_ds, dest_speaker = speaker.split(',')
  for ds, speaker, speaker_id in ds_speakers:
    if speaker == dest_speaker and ds == dest_ds:
      final_speaker_id = speaker_id
      break
  if final_speaker_id == -1:
    raise Exception("Speaker {} not available!".format(speaker))

  for i, serialized_symbol_ids in tqdm(enumerate(serialized_symbol_ids_sentences), total=len(serialized_symbol_ids_sentences)):
    #print(sentence_symbols)
    symbol_ids = deserialize_symbol_ids(serialized_symbol_ids)
    print("{} ({})".format(conv.ids_to_text(symbol_ids), len(symbol_ids)))
    synthesized_sentence = synt.infer(symbol_ids, final_speaker_id)
    output = np.concatenate((output, synthesized_sentence, sentence_pause_samples), axis=0)
    #print(output)

  print("Saving...")
  last_dir_name = Path(infer_dir_path).parts[-1]
  output_name = "{}.wav".format(last_dir_name)
  out_path = os.path.join(infer_dir_path, output_name)
  to_wav(out_path, output, hparams.sampling_rate)
  print("Finished. Saved to:", out_path)
  print("Plotting...")
  wav = get_audio(out_path)
  #wav = get_segment(wav)
  mel = plotter.get_mel(wav)
  plot_melspecs([mel])
  output_name = "{}.png".format(last_dir_name)
  out_path = os.path.join(infer_dir_path, output_name)
  plt.savefig(out_path, bbox_inches='tight')
  plt.show()


# if __name__ == "__main__":
#   parser = argparse.ArgumentParser()
#   parser.add_argument('--base_dir', type=str, help='base directory')
#   parser.add_argument('--checkpoint', type=str, help='checkpoint name')
#   parser.add_argument('--output_name', type=str, help='name of the wav file', default='complete')
#   parser.add_argument('--waveglow', type=str, help='Path to pretrained waveglow file')
#   parser.add_argument('--hparams', type=str, required=False, help='comma separated name=value pairs')
#   parser.add_argument('--ds_name', type=str, required=False)
#   parser.add_argument('--speaker', type=str, required=False)
#   parser.add_argument('--debug', type=str, default='true')

#   = parser.parse_)
#   hparams = create_hparams(hparams)
#   debug = str.lower(debug) == 'true'
#   if debug:
#     base_dir = '/datasets/models/taco2pt_ms'
#     speaker_dir = os.path.join(base_dir, filelist_dir)
#     checkpoint_path = os.path.join(base_dir, savecheckpoints_dir, 'ljs_ipa_thchs_no_tone_A11_1499')
#     #checkpoint_path = os.path.join(base_dir, checkpoint_output_dir, 'checkpoint_1499')
#     waveglow = '/datasets/models/pretrained/waveglow_256channels_universal_v5.pt'
#     output_name = 'test'
#     hparams.sampling_rate = 19000
#   else:
#     speaker_dir = os.path.join(base_dir, filelist_dir, ds_name, speaker)
#     checkpoint_path = os.path.join(base_dir, savecheckpoints_dir, checkpoint)
