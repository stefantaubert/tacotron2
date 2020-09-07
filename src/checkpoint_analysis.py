import os
import pickle

import torch


def load_checkpoint(checkpoint_path):
  assert os.path.isfile(checkpoint_path)
  with open(checkpoint_path, "r", encoding="unicode") as f:
    x = pickle.load(f)
  print("Loading checkpoint '{}'".format(checkpoint_path))
  checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
  for k, v in checkpoint_dict.items():
    print(k)
    # print(v)

  state_dict_key = 'state_dict'
  if state_dict_key in checkpoint_dict:
    print('============')
    print(state_dict_key, 'keys:')
    for k, v in checkpoint_dict[state_dict_key].items():
      print(k)

    print('============')
    print('state_dict.embedding.weights')
    w = checkpoint_dict['state_dict']['embedding.weight']
    print(w.shape)

  optimizer_key = 'optimizer'
  if optimizer_key in checkpoint_dict:
    print('============')
    print(optimizer_key, 'keys:')
    for k, v in checkpoint_dict[optimizer_key].items():
      print(k, type(v))
      for k2, v2 in v.items():
        print(k2, type(v))

  # model.load_state_dict(checkpoint_dict['state_dict'])
  # optimizer.load_state_dict(checkpoint_dict['optimizer'])
  #learning_rate = checkpoint_dict['learning_rate']
  #iteration = checkpoint_dict['iteration']
  #print("Loaded checkpoint '{}' from iteration {}" .format(checkpoint_path, iteration))
  # return model, optimizer, learning_rate, iteration


if __name__ == "__main__":
  # load_checkpoint('/datasets/models/pretrained/tacotron2_statedict.pt')
  # load_checkpoint('/datasets/models/pretrained/ljs_ipa_scratch_80000')
  load_checkpoint('/datasets/phil_home/taco2pt_v2/pretrained/waveglow_256channels_universal_v5.pt')
