import os
import shutil
import sys
import tempfile

import torch



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

  res = {}
  if "iteration" in checkpoint_dict.keys():
    res["iteration"] = checkpoint_dict["iteration"]

  if "optimizer" in checkpoint_dict.keys():
    res["optimizer"] = checkpoint_dict["optimizer"]

  if "learning_rate" in checkpoint_dict.keys():
    res["learning_rate"] = checkpoint_dict["learning_rate"]

  if "model" in checkpoint_dict.keys():
    model = checkpoint_dict["model"]
    res["state_dict"] = model.state_dict()

  torch.save(res, destination)
  print("Successfully converted. Output:", destination)


if __name__ == "__main__":
  _convert_core(
    source="/datasets/tmp/wg_v3/v3.pt",
    destination="/datasets/tmp/wg_v3/v3_conv.pt"
  )

  convert_glow(
    origin='/datasets/tmp/wg_v3/v3.pt',
    destination='/datasets/tmp/wg_v3/v3_conv.pt',
    keep_orig=True
  )
