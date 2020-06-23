import sys

import argparse
import matplotlib
import matplotlib.pylab as plt
import numpy as np
from scipy.io import wavfile
import time

import os
import torch
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from text.symbol_converter import load_from_file, deserialize_symbol_ids
from paths import get_symbols_path, inference_input_symbols_file_name, inference_output_file_name, get_checkpoint_dir
from train import get_last_checkpoint
from utils import parse_ds_speakers

# to load denoiser, glow etc.
sys.path.append('waveglow/')
from denoiser import Denoiser

from hparams import create_hparams
from model import Tacotron2
from train import load_model
from scipy.io.wavfile import write

def plot_data(data, figsize=(16, 4)):
  fig, axes = plt.subplots(1, len(data), figsize=figsize)
  for i in range(len(data)):
    axes[i].imshow(data[i], aspect='auto', origin='bottom', interpolation='none')
  ### todo path
  plt.savefig("out/plot.png", bbox_inches='tight')

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

  def infer(self, symbols, dest_name, speaker_id: int):
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

def infer(training_dir_path: str, infer_dir_path: str, config: dict, train_config: dict):
  hparams = create_hparams(config["hparams"])

  conv = load_from_file(get_symbols_path(training_dir_path))
  n_symbols = conv.get_symbol_ids_count()
  print('Loaded {} symbols'.format(n_symbols))
  ds_speakers = parse_ds_speakers(train_config["speakers"])
  n_speakers = len(ds_speakers)
  print('Loaded {} speakers'.format(n_speakers))

  with open(os.path.join(infer_dir_path, inference_input_symbols_file_name), 'r') as f:
    serialized_symbol_ids_sentences = f.readlines()

  hparams.n_symbols = n_symbols
  hparams.n_speakers = n_speakers

  checkpoint = get_last_checkpoint(training_dir_path)
  if config["custom_checkpoint"] != '':
    checkpoint = config["custom_checkpoint"]
  checkpoint_path = os.path.join(get_checkpoint_dir(training_dir_path), checkpoint)
  print("Using model:", checkpoint_path)
  synt = Synthesizer(hparams, checkpoint_path, config["waveglow"])

  #complete_text = [item for sublist in sentences_symbols for item in sublist]
  #print(complete_text)
  #res = synt.infer(complete_text, "aei")
  #to_wav("out/complete_x.wav", res, synt.hparams.sampling_rate)
  #print("exit")

  # Speed is: 1min inference for 3min wav result

  sentence_pause = np.zeros(10**4)

  print("Inferring...")

  output = np.array([])

  final_speaker_id = -1
  dest_ds, dest_speaker = config["speaker"].split(',')
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
    synthesized_sentence = synt.infer(symbol_ids, str(i), final_speaker_id)
    output = np.concatenate((output, synthesized_sentence, sentence_pause), axis=0)
    #print(output)

  print("Saving...")
  out_path = os.path.join(infer_dir_path, inference_output_file_name)
  to_wav(out_path, output, synt.hparams.sampling_rate)
  print("Finished. Saved to:", out_path)


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

#   args = parser.parse_args()
#   hparams = create_hparams(args.hparams)
#   debug = str.lower(args.debug) == 'true'
#   if debug:
#     args.base_dir = '/datasets/models/taco2pt_ms'
#     speaker_dir = os.path.join(args.base_dir, filelist_dir)
#     checkpoint_path = os.path.join(args.base_dir, savecheckpoints_dir, 'ljs_ipa_thchs_no_tone_A11_1499')
#     #checkpoint_path = os.path.join(args.base_dir, checkpoint_output_dir, 'checkpoint_1499')
#     args.waveglow = '/datasets/models/pretrained/waveglow_256channels_universal_v5.pt'
#     args.output_name = 'test'
#     hparams.sampling_rate = 19000
#   else:
#     speaker_dir = os.path.join(args.base_dir, filelist_dir, args.ds_name, args.speaker)
#     checkpoint_path = os.path.join(args.base_dir, savecheckpoints_dir, args.checkpoint)
