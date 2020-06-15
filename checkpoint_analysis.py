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
      
  #model.load_state_dict(checkpoint_dict['state_dict'])
  #optimizer.load_state_dict(checkpoint_dict['optimizer'])
  #learning_rate = checkpoint_dict['learning_rate']
  #iteration = checkpoint_dict['iteration']
  #print("Loaded checkpoint '{}' from iteration {}" .format(checkpoint_path, iteration))
  #return model, optimizer, learning_rate, iteration

if __name__ == "__main__":
  #load_checkpoint('/datasets/models/pretrained/tacotron2_statedict.pt')
  load_checkpoint('/datasets/models/taco2pt_ms/saved_checkpoints/ljs_1_ipa_49000')