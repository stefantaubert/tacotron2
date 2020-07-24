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
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDwav_tensor, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************
import os
import random
import argparse
import json
import torch
import torch.utils.data
import sys
from scipy.io.wavfile import read

from src.tacotron.layers import TacotronSTFT
from src.waveglow.prepare_ds import load_filepaths
from src.common.audio.utils import wav_to_float32

class MelParser():
  def __init__(self, hparams):
    super().__init__()
    self.stft = TacotronSTFT(
      filter_length=hparams.filter_length,
      hop_length=hparams.hop_length,
      win_length=hparams.win_length,
      n_mel_channels=hparams.n_mel_channels,
      sampling_rate=hparams.sampling_rate,
      mel_fmin=hparams.mel_fmin,
      mel_fmax=hparams.mel_fmax
    )

  def get_mel(self, path, segment_length=None):
    wav, sampling_rate = wav_to_float32(path)

    if sampling_rate != self.stft.sampling_rate:
      raise ValueError("{} {} SR doesn't match target {} SR".format(filename, sampling_rate, self.stft.sampling_rate))
    
    wav_tensor = torch.FloatTensor(wav)

    if segment_length:
      # Take segment
      if wav_tensor.size(0) >= segment_length:
        max_audio_start = wav_tensor.size(0) - segment_length
        audio_start = random.randint(0, max_audio_start)
        wav_tensor = wav_tensor[audio_start:audio_start+segment_length]
      else:
        wav_tensor = torch.nn.functional.pad(wav_tensor, (0, segment_length - wav_tensor.size(0)), 'constant').data
    
    mel = self.__get_mel_core(wav_tensor)

    return (mel, wav_tensor)
      
  def __get_mel_core(self, wav_tensor):
    wav_tensor = wav_tensor.unsqueeze(0)
    wav_tensor = torch.autograd.Variable(wav_tensor, requires_grad=False)
    melspec = self.stft.mel_spectrogram(wav_tensor)
    melspec = torch.squeeze(melspec, 0)
    return melspec

class Mel2Samp(torch.utils.data.Dataset):
  """
  This is the main class that calculates the spectrogram and returns the
  spectrogram, audio pair.
  """
  def __init__(self, training_files, hparams):
    self.audio_files = load_filepaths(training_files)
    random.seed(hparams.seed)
    random.shuffle(self.audio_files)
    self.mel_parser = MelParser(hparams)
    self.segment_length = hparams.segment_length

  def __getitem__(self, index):
    filename = self.audio_files[index][0]
    mel, wav_tensor = self.mel_parser.get_mel(filename, segment_length=self.segment_length)
    return (mel, wav_tensor)

  def __len__(self):
    return len(self.audio_files)

# # ===================================================================
# # Takes directory of clean audio and makes directory of spectrograms
# # Useful for making test sets
# # ===================================================================
# if __name__ == "__main__":
#   # Get defaults so it can work with no Sacred
#   parser = argparse.ArgumentParser()
#   parser.add_argument('-f', "--filelist_path", required=True)
#   parser.add_argument('-c', '--config', type=str,
#             help='JSON file for configuration')
#   parser.add_argument('-o', '--output_dir', type=str,
#             help='Output directory')
#   args = parser.parse_args()

#   with open(args.config) as f:
#     data = f.read()
#   data_config = json.loads(data)["data_config"]
#   mel2samp = Mel2Samp(**data_config)

#   filepaths = files_to_list(args.filelist_path)

#   # Make directory if it doesn't exist
#   if not os.path.isdir(args.output_dir):
#     os.makedirs(args.output_dir)
#     os.chmod(args.output_dir, 0o775)

#   for filepath in filepaths:
#     audio, sr = load_wav_to_torch(filepath)
#     melspectrogram = mel2samp.get_mel(audio)
#     filename = os.path.basename(filepath)
#     new_filepath = args.output_dir + '/' + filename + '.pt'
#     print(new_filepath)
#     torch.save(melspectrogram, new_filepath)
