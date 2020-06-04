import sys

import matplotlib
import matplotlib.pylab as plt
import numpy as np
from scipy.io import wavfile

import torch

# to load denoiser, glow etc.
sys.path.append('waveglow/')
from denoiser import Denoiser

from hparams import create_hparams
from model import Tacotron2
from text import text_to_sequence
from train import load_model
from scipy.io.wavfile import write

def plot_data(data, figsize=(16, 4)):
  fig, axes = plt.subplots(1, len(data), figsize=figsize)
  for i in range(len(data)):
    axes[i].imshow(data[i], aspect='auto', origin='bottom', interpolation='none')
  plt.savefig("out/plot.png", bbox_inches='tight')

def to_wav(path, data, sr):
  wav = data
  wav *= (2**15 - 1) / max(10**-2, np.max(np.abs(wav)))
  #wavfile.write(path, rate=sr, data=wav.astype(np.int16))
  write(path, sr, wav.astype(np.int16))

class Synthesizer():
  def __init__(self):
    super().__init__()
    self.hparams = create_hparams()
    self.hparams.sampling_rate = 22050

    # Load model from checkpoint
    checkpoint_path = "pretrained/tacotron2_statedict.pt"
    #checkpoint_path = "/datasets/models/taco2pytorch/checkpoint_10"
    self.model = load_model(self.hparams)
    self.model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    self.model.cuda().eval().half()

    # Load WaveGlow for mel2audio synthesis and denoiser
    waveglow_path = 'pretrained/waveglow_256channels_universal_v5.pt'
    self.waveglow = torch.load(waveglow_path)['model']
    self.waveglow.cuda().eval().half()
    for k in self.waveglow.convinv:
      k.float()
    self.denoiser = Denoiser(self.waveglow)

  def infer(self, text, dest_name):
    sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()

    # Decode text input and plot results
    mel_outputs, mel_outputs_postnet, _, alignments = self.model.inference(sequence)
    # plot_data((mel_outputs.float().data.cpu().numpy()[0], mel_outputs_postnet.float().data.cpu().numpy()[0], alignments.float().data.cpu().numpy()[0].T))

    with torch.no_grad():
      audio = self.waveglow.infer(mel_outputs_postnet, sigma=0.666)
    res = audio[0].data.cpu().numpy()
    #print("Saving {}...".format(dest_name))
    #to_wav("out/{}.wav".format(dest_name), res, self.hparams.sampling_rate)
    return res

    # (Optional) Remove WaveGlow bias
    audio_denoised = self.denoiser(audio, strength=10**-2)[:, 0]
    print("Saving...")
    to_wav("out/{}_denoised.wav".format(dest_name), audio_denoised.cpu().numpy(), self.hparams.sampling_rate)

if __name__ == "__main__":
  from tqdm import tqdm
  from nltk.tokenize import sent_tokenize

  output = np.array([])
  lines = []

  with open('in/text.txt', 'r') as f:
    lines = f.readlines()
  sentences = []
  for line in lines:
    sents = sent_tokenize(line)
    sentences.extend(sents)

  print('\n'.join(sentences))
  
  # Speed is: 1min inference for 3min wav result
  synt = Synthesizer()
  for i, line in tqdm(enumerate(sentences), total=len(sentences)):
    #print("Inferring...", line)
    res = synt.infer(line, str(i))
    output = np.concatenate((output, res), axis=0)
    #print(output)

  print("Saving...")
  to_wav("out/complete.wav", output, synt.hparams.sampling_rate)

    