import os
import torch
from src.core.common.train_log import log
from src.core.common.utils import get_subdir
from shutil import copyfile
import datetime
from src.core.common.utils import args_to_str, get_last_checkpoint, parse_json

checkpoint_dir = 'checkpoints'

train_config_file = 'config.log'
description_txt_file = 'description.txt'

def init_path_parser(parser):
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--custom_training_name', type=str)
  return main

def create_description_file(training_dir_path: str):
  desc_file = os.path.join(training_dir_path, description_txt_file)
  if not os.path.exists(desc_file):
    with open(desc_file, 'w', encoding='utf-8') as f:
      f.write('Description\n')
      f.write('-----------\n')

def log_train_config(training_dir_path: str, args):
  t = args_to_str(args)
  with open(os.path.join(training_dir_path, train_config_file), 'w', encoding='utf-8') as f:
    f.write(t)
  print("Passed training arguments:")
  print(t)

def main(base_dir: str, custom_training_name: str):
  #print(base_dir)
  #print(custom_training_name)
  if custom_training_name != None and custom_training_name != "":
    train_dir_path = os.path.join(base_dir, custom_training_name)
    os.makedirs(train_dir_path, exist_ok=True)
    print("Ensured folder {} exists.".format(train_dir_path))
    train_dir_name = custom_training_name
  else:
    train_dir_name = get_training_dir(base_dir, create=True)
    train_dir_path = os.path.join(base_dir, train_dir_name)

  create_description_file(train_dir_path)

  print(train_dir_name)

def get_checkpoint(training_dir_path: str, custom_checkpoint: int):
  if custom_checkpoint:
    checkpoint = custom_checkpoint
  else:
    checkpoint_dir = get_checkpoint_dir(training_dir_path)
    checkpoint = get_last_checkpoint(checkpoint_dir)
  checkpoint_path = os.path.join(get_checkpoint_dir(training_dir_path), str(checkpoint))
  return checkpoint, checkpoint_path

def get_training_dir(base_dir: str, create: bool = True) -> str:
  training_dir = "training_{}".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
  training_dir_path = os.path.join(base_dir, training_dir)
  if create:
    os.makedirs(training_dir_path, exist_ok=True)
  return training_dir

def get_checkpoint_dir(training_dir_path: str, create: bool = True) -> str:
  return get_subdir(training_dir_path, checkpoint_dir, create)

def get_continue_training_model_checkpoint(training_dir_path: str):
  checkpoint_dir = get_checkpoint_dir(training_dir_path)
  last_checkpoint = get_last_checkpoint(checkpoint_dir)

  if not last_checkpoint:
    raise Exception("No checkpoint was found to continue training!")

  full_checkpoint_path = os.path.join(get_checkpoint_dir(training_dir_path), last_checkpoint)
  return full_checkpoint_path

def load_checkpoint(checkpoint_path, model, optimizer, training_dir_path):
  #weights_path = os.path.join(speaker_dir, weights_name)
  #assert os.path.isfile(weights_path)
  assert os.path.isfile(checkpoint_path)
  checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
  ### Didn't worked out bc the optimizer has old weight size
  # if overwrite_weights:
  #   weights = np.load(weights_path)
  #   weights = torch.from_numpy(weights)
  #   dummy_dict = model.state_dict()
  #   update = { 
  #       'embedding.weight': weights 
  #   }
  #   checkpoint_dict.update({'iteration':0})
  #   y_ref = weights[0]
  #   x = checkpoint_dict['state_dict']['embedding.weight'][0]
  #   checkpoint_dict['state_dict'].update(update)
  #   y = checkpoint_dict['state_dict']['embedding.weight'][0]
  #   #checkpoint_dict['state_dict']['embedding.weights'] = weights
  model.load_state_dict(checkpoint_dict['state_dict'])
  optimizer.load_state_dict(checkpoint_dict['optimizer'])
  learning_rate = checkpoint_dict['learning_rate']
  iteration = checkpoint_dict['iteration']
  return model, optimizer, learning_rate, iteration

def save_checkpoint(model, optimizer, learning_rate, iteration, training_dir_path):
  filepath = os.path.join(get_checkpoint_dir(training_dir_path), str(iteration))
  log(training_dir_path, "Saving model and optimizer state at iteration {} to {}".format(iteration, filepath))
  torch.save(
    {
      'iteration': iteration,
      'state_dict': model.state_dict(),
      'optimizer': optimizer.state_dict(),
      'learning_rate': learning_rate
    },
    filepath
  )

def save_checkpoint_score(training_dir_path, iteration, gradloss, trainloss, valloss, epoch, i):
  filepath = os.path.join(get_checkpoint_dir(training_dir_path), str(iteration))
  loss_avg = (trainloss + valloss) / 2
  name = "{}_epoch-{}_it-{}_grad-{:.6f}_train-{:.6f}_val-{:.6f}_avg-{:.6f}.log".format(filepath, epoch, i, gradloss, trainloss, valloss, loss_avg)
  with open(name, mode='w', encoding='utf-8') as f:
    f.write("Training Grad Norm: {:.6f}\nTraining Loss: {:.6f}\nValidation Loss: {:.6f}".format(gradloss, trainloss, valloss))


if __name__ == "__main__":
  main(
    base_dir='/datasets/models/taco2pt_v2',
    custom_training_name='debug_ljs_ms_test'
  )
