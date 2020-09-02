import os
from typing import List, Tuple

PYTORCH_EXT = ".pt"


def get_pytorch_filename(name: str) -> str:
  return f"{name}{PYTORCH_EXT}"


def get_pytorch_basename(filename: str):
  return filename[:-len(PYTORCH_EXT)]


def is_pytorch_file(filename: str):
  return filename.endswith(PYTORCH_EXT)


def get_last_checkpoint(checkpoint_dir: str) -> Tuple[str, int]:
  #checkpoint_dir = get_checkpoint_dir(training_dir_path)
  its = get_all_checkpoint_iterations(checkpoint_dir)
  at_least_one_checkpoint_exists = len(its) > 0
  if not at_least_one_checkpoint_exists:
    raise Exception("No checkpoint iteration found!")
  last_iteration = max(its)
  last_checkpoint = get_pytorch_filename(last_iteration)
  checkpoint_path = os.path.join(checkpoint_dir, last_checkpoint)
  return checkpoint_path, last_iteration


def get_all_checkpoint_iterations(checkpoint_dir: str) -> List[int]:
  _, _, filenames = next(os.walk(checkpoint_dir))
  checkpoints_str = [get_pytorch_basename(x)
                     for x in filenames if is_pytorch_file(x)]
  checkpoints = list(sorted(list(map(int, checkpoints_str))))
  return checkpoints


def get_custom_checkpoint(checkpoint_dir: str, custom_iteration: int) -> Tuple[str, int]:
  checkpoint_path = os.path.join(
    checkpoint_dir, get_pytorch_filename(custom_iteration))
  if not os.path.isfile(checkpoint_path):
    raise Exception(f"Checkpoint with iteration {custom_iteration} not found!")
  return checkpoint_path, custom_iteration


def get_custom_or_last_checkpoint(checkpoint_dir: str, custom_iteration: int) -> Tuple[str, int]:
  return get_custom_checkpoint(checkpoint_dir, custom_iteration) if custom_iteration else get_last_checkpoint(checkpoint_dir)
