import torch
import os
from src.core.common import get_last_checkpoint
from src.core.common import get_subdir

checkpoint_dir = 'checkpoints'

def get_checkpoint(training_dir_path: str, custom_checkpoint):
  if custom_checkpoint:
    checkpoint = custom_checkpoint
  else:
    checkpoint_dir = get_checkpoint_dir(training_dir_path)
    checkpoint = get_last_checkpoint(checkpoint_dir)
  checkpoint_path = os.path.join(get_checkpoint_dir(training_dir_path), checkpoint)
  return checkpoint, checkpoint_path

def get_checkpoint_dir(training_dir_path: str, create: bool = True) -> str:
  return get_subdir(training_dir_path, checkpoint_dir, create)

def get_last_checkpoint_path(training_dir_path: str):
  checkpoint_dir = get_checkpoint_dir(training_dir_path)
  last_checkpoint = get_last_checkpoint(checkpoint_dir)

  if not last_checkpoint:
    raise Exception("No checkpoint was found to continue training!")

  full_checkpoint_path = os.path.join(checkpoint_dir, last_checkpoint)
  return full_checkpoint_path