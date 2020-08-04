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
from src.common.audio.utils import (get_wav_tensor_segment,
                                    wav_to_float32_tensor)
from src.pre.mel_parser import MelParser
from src.waveglow.prepare_ds_io import get_wavepath


class MelLoader(torch.utils.data.Dataset):
  """
  This is the main class that calculates the spectrogram and returns the
  spectrogram, audio pair.
  """
  def __init__(self, prepare_ds_data, hparams):
    self.segment_length = hparams.segment_length
    self.mel_parser = MelParser(hparams)

    data = prepare_ds_data
    random.seed(hparams.seed)
    random.shuffle(data)

    print("Loading wavs into memory...")
    self.cache = {}
    for i, values in tqdm(enumerate(data), total=len(data)):
      wav_path = get_wavepath(values)
      wav_tensor, sr = wav_to_float32_tensor(wav_path)
      if sr != hparams.sampling_rate:
        raise ValueError("{} {} SR doesn't match target {} SR".format(wav_path, sr, hparams.sampling_rate))
      self.cache[i] = wav_tensor
    print("Done")

  def __getitem__(self, index):
    wav_tensor = self.cache[index].clone().detach()
    wav_tensor = get_wav_tensor_segment(wav_tensor, self.segment_length)
    mel_tensor = self.mel_parser.get_mel_tensor(wav_tensor)
    return (mel_tensor, wav_tensor)

  def __len__(self):
    return len(self.cache)
