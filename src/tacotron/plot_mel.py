import os
import random
import sys

import matplotlib
import matplotlib.pylab as plt
import numpy as np
from scipy.io.wavfile import read
from tqdm import tqdm

import torch
from src.common.utils import load_filepaths_and_symbols
from src.tacotron.hparams import create_hparams
from src.tacotron.layers import TacotronSTFT
from PIL import Image

def plot_melspec(mel, mel_dim_x=16, mel_dim_y=5, factor=1, title=None):
  height, width = mel.shape
  width_factor = width / 1000
  fig, axes = plt.subplots(
    nrows=1,
    ncols=1,
    figsize=(mel_dim_x*factor*width_factor, mel_dim_y*factor),
  )

  axes.set_title(title)
  axes.set_yticks(np.arange(0, height, step=10))
  axes.set_xticks(np.arange(0, width, step=100))
  axes.set_xlabel("Time (ms)")
  axes.set_ylabel("Freq. channel")
  axes.imshow(mel, aspect='auto', origin='bottom', interpolation='none')
  return axes
  
# def plot_melspecs(melspecs: list, mel_dim_x=16, mel_dim_y=5, factor=1, titles=None) -> None:
#   witdh = melspecs[0].shape[1]
#   fig, axes = plt.subplots(
#     nrows=len(melspecs),
#     ncols=1,
#     figsize=(mel_dim_x*factor*witdh/1000, len(melspecs)*mel_dim_y*factor),
#     # gridspec_kw={
#     #   'height_ratios': [1, 0.5]
#     # }
#   )
#   for i, mel in enumerate(melspecs):
#     witdh = mel.shape[1]
#     dest = axes
#     if len(melspecs) > 1:
#       dest = dest[i]
#     if titles:
#       dest.set_title(titles[i])
#     dest.imshow(mel, aspect='auto', origin='bottom', interpolation='none')

def stack_images_vertically(list_im, out_path):
  images = [Image.open(i) for i in list_im]
  widths, heights = zip(*(i.size for i in images))

  total_height = sum(heights)
  max_width = max(widths)

  new_im = Image.new(
    mode='RGB',
    size=(max_width, total_height),
    color=(255, 255, 255) # white
  )

  y_offset = 0
  for im in images:
    new_im.paste(im, (0, y_offset))
    y_offset += im.size[1]
  new_im.save(out_path)

#if __name__ == "__main__":
  # hparams = create_hparams()

  # mel2samp = Mel2Samp(hparams)
  
  # wav_paths = [
  #   "/datasets/Report/21.07.20/exp1/validation_2020-07-21_09-27-29_D31_888_D31_10500 (very good)/validation_2020-07-21_09-27-29_D31_888_D31_10500_orig.wav",
  #   "/datasets/Report/21.07.20/exp1/validation_2020-07-21_09-27-29_D31_888_D31_10500 (very good)/validation_2020-07-21_09-27-29_D31_888_D31_10500_inferred.wav",
  #   #"/datasets/thchs_16bit_22050kHz/wav/train/C18/C18_742.wav",
  #   #"/datasets/thchs_16bit_22050kHz/wav/train/C18/C18_743.wav",
  #   #"/datasets/thchs_16bit_22050kHz/wav/train/C18/C18_744.wav"
  # ]

  # melspectrograms = []
  # for i, w in enumerate(wav_paths):
  #   audio = get_audio(w, target_sr=hparams.sampling_rate)
  #   #audio = get_segment(audio)
  #   melspectrogram = mel2samp.get_mel(audio)
  #   #new_filepath = "/tmp/out.pt"
  #   #torch.save(melspectrogram, new_filepath)
  #   melspectrograms.append(melspectrogram)

  #   plot_melspec(melspectrogram, mel_dim_x = 16, factor=1, title=str(i))
  #   plt.savefig("/tmp/plot{}.png".format(i), bbox_inches='tight')
  #   plt.show()

  # stack_images_vertically([
  #   "/tmp/plot0.png",
  #   "/tmp/plot1.png"
  # ],
  #   "/tmp/plot.png"
  # )
  # #plot_melspecs(melspectrograms, mel_dim_x = 4, factor=1, titles=["a", "b"])
  # #plt.savefig("/tmp/plot.png", bbox_inches='tight')
  

  # plt.show()
