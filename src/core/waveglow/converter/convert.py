import os

import torch


def convert_main(source: str, destination: str):
  assert os.path.isfile(source)
  checkpoint_dict = torch.load(source, map_location='cpu')
  res = {}
  if "iteration" in checkpoint_dict:
    res["iteration"] = checkpoint_dict["iteration"]

  if "optimizer" in checkpoint_dict:
    res["optimizer"] = checkpoint_dict["optimizer"]

  if "learning_rate" in checkpoint_dict:
    res["learning_rate"] = checkpoint_dict["learning_rate"]

  if "model" in checkpoint_dict:
    model = checkpoint_dict["model"]
    res["state_dict"] = model.state_dict()

  torch.save(res, destination)
  print("Successfully converted. Output:", destination)

if __name__ == "__main__":
  convert_main(
    source = '/datasets/phil_home/taco2pt_v2/pretrained/waveglow_256channels_universal_v5.pt',
    destination = '/datasets/phil_home/taco2pt_v2/pretrained/waveglow_256channels_universal_v5_out.pt'
  )
