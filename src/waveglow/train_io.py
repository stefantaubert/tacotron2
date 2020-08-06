import torch
import os
from src.common.utils import get_last_checkpoint

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

def save_checkpoint(model, optimizer, learning_rate, iteration, filepath, hparams):
  print("Saving model and optimizer state at iteration {} to {}".format(iteration, filepath))
  #model_for_saving = WaveGlow(hparams).cuda()
  #model_for_saving.load_state_dict(model.state_dict())

  data = {
    #'model': model_for_saving,
    'state_dict': model.state_dict(),
    'iteration': iteration,
    'optimizer': optimizer.state_dict(),
    'learning_rate': learning_rate
  }

  torch.save(data, filepath)

def get_last_checkpoint_path(training_dir_path: str):
  checkpoint_dir = get_checkpoint_dir(training_dir_path)
  last_checkpoint = get_last_checkpoint(checkpoint_dir)

  if not last_checkpoint:
    raise Exception("No checkpoint was found to continue training!")

  full_checkpoint_path = os.path.join(checkpoint_dir, last_checkpoint)
  return full_checkpoint_path