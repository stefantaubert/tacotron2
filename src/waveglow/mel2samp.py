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
from tqdm import tqdm

from src.tacotron.layers import TacotronSTFT
from src.waveglow.prepare_ds import load_filepaths
from src.common.audio.utils import wav_to_float32

class MelLoader(torch.utils.data.Dataset):
  """
  This is the main class that calculates the spectrogram and returns the
  spectrogram, audio pair.
  """
  def __init__(self, training_files, hparams):
    audio_files = load_filepaths(training_files)

    random.seed(hparams.seed)
    random.shuffle(audio_files)

    print("Loading mels into memory...")
    self.cache = {}
    for i, data in tqdm(enumerate(audio_files), total=len(audio_files)):
      mel_path = data[0]
      mel_tensor = torch.load(mel_path, map_location='cpu')
      self.cache[i] = (mel_tensor, wav_tensor)

    self.segment_length = hparams.segment_length

  def __getitem__(self, index):
    mel_tensor, wav_tensor = self.cache[index]
    
    mel, wav_tensor = self.mel_parser.get_mel(filename, segment_length=self.segment_length)
    return (mel, wav_tensor)

  def __len__(self):
    return len(self.audio_files)
