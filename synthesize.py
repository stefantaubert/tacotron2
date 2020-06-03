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
  def __init__(self, hparams):
    super().__init__()
    self.hparams = hparams
    # Load model from checkpoint
    #checkpoint_path = "pretrained/tacotron2_statedict.pt"
    checkpoint_path = "/datasets/models/taco2pytorch/checkpoint_5"
    self.model = load_model(self.hparams)
    torch_model = torch.load(checkpoint_path)
    state_dict = torch_model['state_dict']
    self.model.load_state_dict(state_dict)
    self.model.cuda().eval().half()

    # Load WaveGlow for mel2audio synthesis and denoiser
    waveglow_path = 'pretrained/waveglow_256channels_universal_v5.pt'
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
  from text.conversion.SymbolConverter import get_from_file

  conv = get_from_file('/tmp/symbols.json')
  n_symbols = conv.get_symbols_count()

  output = np.array([])

  sentences_symbols = []
  with open('in/text_sents_accented_seq.txt', 'r') as f:
    sentences_symbols = f.readlines()
  sentences_symbols = [x.split(',') for x in sentences_symbols]
  sentences_symbols = [list(map(int, l)) for l in sentences_symbols]

  hparams = create_hparams()
  hparams.sampling_rate = 22050
  hparams.n_symbols = n_symbols
  # Speed is: 1min inference for 3min wav result
  synt = Synthesizer(hparams)
  for i, sentence_symbols in tqdm(enumerate(sentences_symbols), total=len(sentences_symbols)):
    #print("Inferring...", line)
    res = synt.infer(sentence_symbols, str(i))
    output = np.concatenate((output, res), axis=0)
    #print(output)

  print("Saving...")
  to_wav("out/complete.wav", output, synt.hparams.sampling_rate)

    