import torch
import os
from src.core.common.utils import get_last_checkpoint
from src.core.common.utils import get_subdir

checkpoint_dir = 'checkpoints'

def load_checkpoint(checkpoint_path, model, optimizer):
  assert os.path.isfile(checkpoint_path)
  checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
  iteration = checkpoint_dict['iteration']
  learning_rate = checkpoint_dict['learning_rate']
  optimizer_state_dict = checkpoint_dict['optimizer']
  optimizer.load_state_dict(optimizer_state_dict)
  # model_for_loading = checkpoint_dict['model']
  # model.load_state_dict(model_for_loading.state_dict())
  model_state_dict = checkpoint_dict['state_dict']
  model.load_state_dict(model_state_dict)
  print("Loaded checkpoint '{}' (iteration {})" .format(checkpoint_path, iteration))
  return model, optimizer, learning_rate, iteration

def save_checkpoint(training_dir_path, model, optimizer, learning_rate, iteration, hparams):
  checkpoint_dir = get_checkpoint_dir(training_dir_path)
  checkpoint_path = os.path.join(checkpoint_dir,  str(iteration))
  print("Saving model and optimizer state at iteration {} to {}".format(iteration, checkpoint_path))
  #model_for_saving = WaveGlow(hparams).cuda()
  #model_for_saving.load_state_dict(model.state_dict())

  data = {
    #'model': model_for_saving,
    'state_dict': model.state_dict(),
    'iteration': iteration,
    'optimizer': optimizer.state_dict(),
    'learning_rate': learning_rate
  }

  torch.save(data, checkpoint_path)

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