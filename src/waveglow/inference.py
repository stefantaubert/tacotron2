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
import argparse
import os
import tempfile
from pathlib import Path
from shutil import copyfile

import imageio
import matplotlib
import matplotlib.pylab as plt
import numpy as np
from scipy.io.wavfile import write

import torch
from src.common.audio.utils import float_to_wav, is_overamp
from src.common.utils import stack_images_vertically
from src.paths import get_inference_dir
from src.tacotron.layers import TacotronSTFT
from src.pre.mel_parser import MelParser, plot_melspec, compare_mels
from src.waveglow.denoiser import Denoiser
from src.waveglow.hparams import create_hparams
from src.waveglow.train import (get_checkpoint_dir,
                                load_model)
from src.common.utils import get_last_checkpoint



class Synthesizer():
  def __init__(self, checkpoint_path, hparams=None):
    super().__init__()
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model_state_dict = checkpoint_dict['state_dict']
    # TODO pass waveglow hparams in tacotron with arguments (only required if used non default hparams)
    hparams = create_hparams()
    model = load_model(hparams)
    model.load_state_dict(model_state_dict)

    model = model.remove_weightnorm(model)
    model = model.cuda()
    model = model.eval()
    
    denoiser = Denoiser(model)
    denoiser = denoiser.cuda()
    
    self.model = model
    self.denoiser = denoiser

  def infer_mel(self, mel, sigma, denoiser_strength):
    with torch.no_grad():
      audio = self.model.infer(mel, sigma=sigma)
      if denoiser_strength > 0:
        assert self.denoiser
        audio = self.denoiser(audio, denoiser_strength)
    audio = audio.squeeze()
    audio = audio.cpu()
    audio = audio.numpy()
    return audio

def infer(training_dir_path: str, infer_dir_path: str, hparams, checkpoint: str, infer_wav_path: str, denoiser_strength: float, sigma: float):
  hparams = create_hparams(hparams)

  checkpoint_path = os.path.join(get_checkpoint_dir(training_dir_path), str(checkpoint))
  print("Using model:", checkpoint_path)
  synth = Synthesizer(checkpoint_path, hparams)
  
  # if is_fp16:
  #   from apex import amp
  #   waveglow, _ = amp.initialize(waveglow, [], opt_level="O3")

  print("Inferring {}...".format(infer_wav_path))

  mel_parser = MelParser(hparams)
  mel = mel_parser.get_mel(infer_wav_path, segment_length=None)[0]
  mel = mel.cuda()
  mel = torch.autograd.Variable(mel)
  mel = mel.unsqueeze(0)

  #mel = mel.half() if is_fp16 else mel

  audio = synth.infer_mel(mel, sigma, denoiser_strength)

  last_dir_name = Path(infer_dir_path).parts[-1]
  output_name = "{}".format(last_dir_name)
  out_path_template = os.path.join(infer_dir_path, output_name)
  path_original_wav = "{}_orig.wav".format(out_path_template)
  path_original_plot = "{}_orig.png".format(out_path_template)
  path_inferred_wav = "{}_inferred.wav".format(out_path_template)
  path_inferred_plot = "{}_inferred.png".format(out_path_template)
  path_compared_plot = "{}_comparison.png".format(out_path_template)

  assert not is_overamp(audio)
  float_to_wav(
    wav=audio,
    path=path_inferred_wav,
    dtype=np.int16,
    normalize=False,
    sample_rate=hparams.sampling_rate
  )

  print("Plotting...")

  mel_inferred = mel_parser.get_mel(path_inferred_wav)[0]
  mel_orig = mel_parser.get_mel(infer_wav_path)[0]

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

def init_inference_parser(parser):
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--training_dir', type=str, required=True)
  parser.add_argument('--wav', type=str, required=True)
  parser.add_argument('--hparams', type=str)
  parser.add_argument("--denoiser_strength", default=0.0, type=float, help='Removes model bias.')
  parser.add_argument("--sigma", default=1.0, type=float)
  parser.add_argument('--custom_checkpoint', type=int)
  return __main

def __main(base_dir, training_dir, wav, hparams, denoiser_strength, sigma, custom_checkpoint):
  training_dir_path = os.path.join(base_dir, training_dir)

  checkpoint_dir = get_checkpoint_dir(training_dir_path)
  checkpoint = custom_checkpoint if str(custom_checkpoint) else get_last_checkpoint(checkpoint_dir)

  wav_name = os.path.basename(wav)[:-4]
  infer_dir = get_inference_dir(training_dir_path, wav_name, checkpoint, "{}_{}".format(sigma, denoiser_strength))

  infer(
    training_dir_path=training_dir_path,
    infer_dir_path=infer_dir,
    hparams=hparams,
    checkpoint=checkpoint,
    infer_wav_path=wav,
    denoiser_strength=denoiser_strength,
    sigma=sigma
  )


if __name__ == "__main__":
  __main(
    base_dir = '/datasets/models/taco2pt_v2',
    training_dir = 'wg_debug',
    wav = "/datasets/LJSpeech-1.1-test/wavs/LJ001-0100.wav",
    denoiser_strength = 0,
    sigma = 0.666,
    custom_checkpoint = ''
  )
