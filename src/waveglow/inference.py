# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#    * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#    * Neither the name of the NVIDIA CORPORATION nor the
#    names of its contributors may be used to endorse or promote products
#    derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************
import os
from pathlib import Path
from shutil import copyfile
import tempfile

import imageio
import matplotlib
matplotlib.use("Agg")

import matplotlib.pylab as plt
from scipy.io.wavfile import write

import torch
from src.common.utils import compare_mels
from src.tacotron.script_plot_mel import (Mel2Samp, get_audio, get_segment,
                                          plot_melspec,
                                          stack_images_vertically)
from src.waveglow.denoiser import Denoiser
from src.waveglow.hparams import create_hparams
from src.waveglow.mel2samp import MAX_WAV_VALUE
from src.waveglow.train import get_checkpoint_dir, load_model


def infer(training_dir_path: str, infer_dir_path: str, hparams, checkpoint: str, infer_wav_path: str, denoiser_strength: float, sigma: float):
  hparams = create_hparams(hparams)
  plotter = Mel2Samp(hparams)

  checkpoint_path = os.path.join(get_checkpoint_dir(training_dir_path), checkpoint)
  print("Using model:", checkpoint_path)
  assert os.path.isfile(checkpoint_path)
  
  checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
  model_state_dict = checkpoint_dict['state_dict']
  model = load_model(hparams)
  model.load_state_dict(model_state_dict)
  model = model.remove_weightnorm(model)
  model.cuda().eval()

  # if is_fp16:
  #   from apex import amp
  #   waveglow, _ = amp.initialize(waveglow, [], opt_level="O3")

  if denoiser_strength > 0:
    denoiser = Denoiser(model).cuda()

  print("Inferring {}...".format(infer_wav_path))

  a = get_audio(infer_wav_path)
  mel = plotter.get_mel(a)
  
  #mel = torch.load(file_path)
  mel = torch.autograd.Variable(mel.cuda())
  mel = torch.unsqueeze(mel, 0)
  #mel = mel.half() if is_fp16 else mel

  with torch.no_grad():
    audio = model.infer(mel, sigma=sigma)
    if denoiser_strength > 0:
      audio = denoiser(audio, denoiser_strength)
    audio = audio * MAX_WAV_VALUE
  audio = audio.squeeze()
  audio = audio.cpu().numpy()
  audio = audio.astype('int16')

  last_dir_name = Path(infer_dir_path).parts[-1]
  output_name = "{}".format(last_dir_name)
  out_path_template = os.path.join(infer_dir_path, output_name)
  path_original_wav = "{}_orig.wav".format(out_path_template)
  path_original_plot = "{}_orig.png".format(out_path_template)
  path_inferred_wav = "{}_inferred.wav".format(out_path_template)
  path_inferred_plot = "{}_inferred.png".format(out_path_template)
  path_compared_plot = "{}_comparison.png".format(out_path_template)

  write(path_inferred_wav, hparams.sampling_rate, audio)

  print("Plotting...")
  wav_inferred = get_audio(path_inferred_wav)
  wav_orig = get_audio(infer_wav_path)
  #wav = get_segment(wav)
  mel_inferred = plotter.get_mel(wav_inferred)
  mel_orig = plotter.get_mel(wav_orig)

  ax = plot_melspec(mel_inferred, title="Inferred")
  plt.savefig(path_inferred_plot, bbox_inches='tight')

  ax.set_title('')
  #ax.get_xaxis().set_visible(False)
  #ax.get_yaxis().set_visible(False)
  a = tempfile.mktemp(suffix='.png')
  plt.savefig(a, bbox_inches='tight')

  ax = plot_melspec(mel_orig, title="Original")
  plt.savefig(path_original_plot, bbox_inches='tight')

  ax.set_title('')
  #ax.get_xaxis().set_visible(False)
  #ax.get_yaxis().set_visible(False)
  b = tempfile.mktemp(suffix='.png')
  plt.savefig(b, bbox_inches='tight')

  score, diff_img = compare_mels(a, b)
  os.remove(a)
  os.remove(b)
  path_diff_plot = "{}_diff_{:.6}.png".format(out_path_template, score)
  imageio.imsave(path_diff_plot, diff_img)

  print("Similarity of original and inferred mel: {}".format(score))

  stack_images_vertically([path_original_plot, path_inferred_plot, path_diff_plot], path_compared_plot)

  copyfile(infer_wav_path, path_original_wav)
  print("Finished.")
