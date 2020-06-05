import sys

import argparse
import matplotlib
import matplotlib.pylab as plt
import numpy as np
from scipy.io import wavfile

from paths import checkpoint_output_dir, pretrained_dir, input_symbols, output_dir
import os
import torch
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from text.conversion.SymbolConverter import get_from_file
from script_ds_pre import symbols_path

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

  def infer(self, symbols, dest_name):
    sequence = np.array([symbols])
    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()

    # Decode text input and plot results
    mel_outputs, mel_outputs_postnet, _, alignments = self.model.inference(sequence)
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

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--base_dir', type=str, help='base directory', default='/datasets/models/taco2pt_ipa')
  parser.add_argument('--checkpoint', type=str, help='checkpoint subpath', default='checkpoint_1000')
  parser.add_argument('--output_name', type=str, help='name of the wav file', default='complete')
  
  args = parser.parse_args()

  conv = get_from_file(os.path.join(args.base_dir, symbols_path))
  n_symbols = conv.get_symbols_count()

  output = np.array([])

  sentences_symbols = []
  with open(os.path.join(args.base_dir, input_symbols), 'r') as f:
    sentences_symbols = f.readlines()
  sentences_symbols = [x.split(',') for x in sentences_symbols]
  sentences_symbols = [list(map(int, l)) for l in sentences_symbols]

  hparams = create_hparams()
  hparams.sampling_rate = 22050
  hparams.n_symbols = n_symbols

  #checkpoint_path = os.path.join(args.base_dir, pretrained_dir, 'tacotron2_statedict.pt')
  checkpoint_path = os.path.join(args.base_dir, checkpoint_output_dir, args.checkpoint)
  waveglow_path = os.path.join(args.base_dir, pretrained_dir, 'waveglow_256channels_universal_v5.pt')

  synt = Synthesizer(hparams, checkpoint_path, waveglow_path)
  
  #complete_text = [item for sublist in sentences_symbols for item in sublist]
  #print(complete_text)
  #res = synt.infer(complete_text, "aei")
  #to_wav("out/complete_x.wav", res, synt.hparams.sampling_rate)
  #print("exit")
  
  # Speed is: 1min inference for 3min wav result
  for i, sentence_symbols in tqdm(enumerate(sentences_symbols), total=len(sentences_symbols)):
    #print("Inferring...", line)
    #print(sentence_symbols)
    res = synt.infer(sentence_symbols, str(i))
    output = np.concatenate((output, res), axis=0)
    sentence_pause = np.zeros(10**4)
    output = np.concatenate((output, sentence_pause), axis=0)
    #print(output)

  print("Saving...")
  to_wav(os.path.join(args.base_dir, output_dir, args.output_name + ".wav"), output, synt.hparams.sampling_rate)

    