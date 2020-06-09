import torch 
import os

def load_checkpoint(checkpoint_path):
  assert os.path.isfile(checkpoint_path)
  print("Loading checkpoint '{}'".format(checkpoint_path))
  checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
  for k, v in checkpoint_dict.items():
    print(k)
    #print(v)
  state_dict_key = 'state_dict'
  if state_dict_key in checkpoint_dict:
    print(state_dict_key, 'keys:')
    for k, v in checkpoint_dict[state_dict_key].items():
      print(k)
  w = checkpoint_dict['state_dict']['embedding.weight']
  print(w.shape)
  #model.load_state_dict(checkpoint_dict['state_dict'])
  #optimizer.load_state_dict(checkpoint_dict['optimizer'])
  #learning_rate = checkpoint_dict['learning_rate']
  #iteration = checkpoint_dict['iteration']
  #print("Loaded checkpoint '{}' from iteration {}" .format(checkpoint_path, iteration))
  #return model, optimizer, learning_rate, iteration

if __name__ == "__main__":
  load_checkpoint('/datasets/models/pretrained/tacotron2_statedict.pt')
  #load_checkpoint('/datasets/models/taco2pt_ipa_chn_15500/output/checkpoint_15500')