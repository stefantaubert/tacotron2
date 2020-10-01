import logging
import os
import shutil
import sys
import tempfile
from dataclasses import asdict

import torch

from src.core.waveglow.hparams import HParams
from src.core.waveglow.train import CheckpointWaveglow


def convert_glow(origin: str, destination: str, keep_orig: bool = False):
  tmp_out = tempfile.mktemp()
  _convert_core(origin, tmp_out)
  if keep_orig:
    if origin == destination:
      original_path = "{}.orig".format(origin)
      shutil.move(origin, original_path)
  else:
    os.remove(origin)
  shutil.move(tmp_out, destination)


def _convert_core(source: str, destination: str):
  '''in version 3 there is only "model"'''
  assert os.path.isfile(source)

  sys.path.append("src/core/waveglow/converter/")
  checkpoint_dict = torch.load(source, map_location='cpu')

  hparams = HParams()

  iteration = 1
  if "iteration" in checkpoint_dict.keys():
    iteration = checkpoint_dict["iteration"]

  optimizer = dict()
  if "optimizer" in checkpoint_dict.keys():
    optimizer = checkpoint_dict["optimizer"]

  learning_rate = hparams.learning_rate
  if "learning_rate" in checkpoint_dict.keys():
    learning_rate = checkpoint_dict["learning_rate"]

  state_dict = dict()
  if "model" in checkpoint_dict.keys():
    model = checkpoint_dict["model"]
    state_dict = model.state_dict()

  res = CheckpointWaveglow(
    hparams=asdict(hparams),
    iteration=iteration,
    learning_rate=learning_rate,
    optimizer=optimizer,
    state_dict=state_dict
  )

  res.save(destination, logging.getLogger())

  print("Successfully converted. Output:", destination)


if __name__ == "__main__":
  _convert_core(
    source="/datasets/tmp/wg_v3/v3.pt",
    destination="/datasets/tmp/wg_v3/v3_conv.pt"
  )

  # convert_glow(
  #   origin='/datasets/tmp/wg_v3/v3.pt',
  #   destination='/datasets/tmp/wg_v3/v3_conv.pt',
  #   keep_orig=True
  # )
