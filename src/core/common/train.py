import os
from typing import Tuple

def get_pytorch_filename(name: str) -> str:
  return f"{name}.pt"

def get_last_checkpoint(checkpoint_dir: str) -> Tuple[str, int]:
  #checkpoint_dir = get_checkpoint_dir(training_dir_path)
  _, _, filenames = next(os.walk(checkpoint_dir))
  filenames = [x[:-3] for x in filenames if ".pt" in x]
  at_least_one_checkpoint_exists = len(filenames) > 0
  if not at_least_one_checkpoint_exists: 
    raise Exception(f"No checkpoint iteration found!")
  iteration = max(list(map(int, filenames)))
  last_checkpoint = get_pytorch_filename(iteration)
  checkpoint_path = os.path.join(checkpoint_dir, last_checkpoint)
  return checkpoint_path, iteration

def get_custom_checkpoint(checkpoint_dir: str, custom_iteration: int) -> Tuple[str, int]:
  checkpoint_path = os.path.join(checkpoint_dir, get_pytorch_filename(custom_iteration))
  if not os.path.isfile(checkpoint_path):
    raise Exception(f"Checkpoint with iteration {custom_iteration} not found!")
  return checkpoint_path, custom_iteration

def get_custom_or_last_checkpoint(checkpoint_dir: str, custom_iteration: int) -> Tuple[str, int]:
  return get_custom_checkpoint(checkpoint_dir, custom_iteration) if custom_iteration else get_last_checkpoint(checkpoint_dir)