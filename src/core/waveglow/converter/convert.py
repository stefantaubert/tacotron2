import logging
import os
import shutil
import sys
import tempfile

import torch

from src.core.common.train import hp_raw
from src.core.waveglow.hparams import create_hparams
from src.core.waveglow.train import Checkpoint


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
  assert os.path.isfile(source)

  sys.path.append("src/core/waveglow/converter/")
  checkpoint_dict = torch.load(source, map_location='cpu')

  hp = create_hparams(None)

  iteration = 1
  if "iteration" in checkpoint_dict.keys():
    iteration = checkpoint_dict["iteration"]

  optimizer = dict()
  if "optimizer" in checkpoint_dict.keys():
    optimizer = checkpoint_dict["optimizer"]

  learning_rate = hp.learning_rate
  if "learning_rate" in checkpoint_dict.keys():
    learning_rate = checkpoint_dict["learning_rate"]

  state_dict = dict()
  if "model" in checkpoint_dict.keys():
    model = checkpoint_dict["model"]
    state_dict = model.state_dict()

  res = Checkpoint(
    hparams=hp_raw(hp),
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
