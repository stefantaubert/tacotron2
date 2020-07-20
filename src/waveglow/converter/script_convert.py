import argparse
import os
import torch

def convert(in_path, out_path):
  assert os.path.isfile(in_path)
  checkpoint_dict = torch.load(in_path, map_location='cpu')
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
  
  torch.save(res, out_path)
  print("Successfully converted. Output:", out_path)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--no_debugging', action='store_true')
  parser.add_argument('--source', type=str)
  parser.add_argument('--destination', type=str)

  args = parser.parse_args()

  if not args.no_debugging:
    args.source = '/datasets/phil_home/taco2pt_v2/pretrained/waveglow_256channels_universal_v5.pt'
    args.destination = '/datasets/phil_home/taco2pt_v2/pretrained/waveglow_256channels_universal_v5_out.pt'

  convert(args.source, args.destination)
