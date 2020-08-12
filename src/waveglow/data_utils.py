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
import random

from tqdm import tqdm

import torch.utils.data
from src.core.common.audio.utils import (get_wav_tensor_segment,
                                    wav_to_float32_tensor)
from src.pre.mel_parser import MelParser
from src.waveglow.prepare_ds_io import PreparedData, PreparedDataList


class MelLoader(torch.utils.data.Dataset):
  """
  This is the main class that calculates the spectrogram and returns the
  spectrogram, audio pair.
  """
  def __init__(self, prepare_ds_data: PreparedDataList, hparams):
    self.mel_parser = MelParser(hparams)
    self.segment_length: int = hparams.segment_length
    self.sampling_rate: int = hparams.sampling_rate
    self.cache_wavs: bool = hparams.cache_wavs

    data = prepare_ds_data
    random.seed(hparams.seed)
    random.shuffle(data)

    wav_paths = {}
    values: PreparedData
    for i, values in enumerate(data):
      wav_paths[i] = values.wav_path
    self.wav_paths = wav_paths

    if hparams.cache_wavs:
      print("Loading wavs into memory...")
      cache = {}
      for i, wav_path in tqdm(wav_paths.items()):
        cache[i] = self.__load_wav_tensor(wav_path)
      print("Done")
      self.cache = cache

  def __load_wav_tensor(self, wav_path: str):
    wav_tensor, sr = wav_to_float32_tensor(wav_path)
    if sr != self.sampling_rate:
      raise ValueError("{} {} SR doesn't match target {} SR".format(wav_path, sr, self.sampling_rate))
    return wav_tensor

  def __getitem__(self, index):
    if self.cache_wavs:
      wav_tensor = self.cache[index].clone().detach()
    else:
      wav_tensor = self.__load_wav_tensor(self.wav_paths[index])
    wav_tensor = get_wav_tensor_segment(wav_tensor, self.segment_length)
    mel_tensor = self.mel_parser.get_mel_tensor(wav_tensor)
    return (mel_tensor, wav_tensor)

  def __len__(self):
    return len(self.wav_paths)
