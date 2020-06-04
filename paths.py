import os
import argparse

filelist_dir = "filelist"
input_dir = "in"
output_dir = "out"
log_dir = 'logs'
checkpoint_output_dir = 'output'
pretrained_dir = 'pretrained'

symbols_path = os.path.join(filelist_dir, 'symbols.json')

training_file = os.path.join(filelist_dir, 'ljs_audio_text_train_filelist.csv')
test_file = os.path.join(filelist_dir, 'ljs_audio_text_test_filelist.csv')
validation_file = os.path.join(filelist_dir, 'ljs_audio_text_val_filelist.csv')
preprocessed_file = os.path.join(filelist_dir, 'ljs_filelist.csv')
preprocessed_file_debug = os.path.join(filelist_dir, 'lj_ipa.csv')

input_text = os.path.join(input_dir, 'text.txt')
input_text_sents = os.path.join(input_dir, 'text_sents.txt')
input_text_sents_accented = os.path.join(input_dir, 'text_sents_accented.txt')
input_symbols = os.path.join(input_dir, 'text_sents_accented_seq.txt')

output_wav = os.path.join(output_dir, 'complete.wav')

checkpoint_file = os.path.join(pretrained_dir, 'tacotron2_statedict.pt')
waveglow_path = os.path.join(pretrained_dir, 'waveglow_256channels_universal_v5.pt')

def ensure_folders_exist(base_dir):
  os.makedirs(os.path.join(base_dir, filelist_dir), exist_ok=True)
  os.makedirs(os.path.join(base_dir, input_dir), exist_ok=True)
  os.makedirs(os.path.join(base_dir, output_dir), exist_ok=True)
  os.makedirs(os.path.join(base_dir, log_dir), exist_ok=True)
  os.makedirs(os.path.join(base_dir, checkpoint_output_dir), exist_ok=True)
  os.makedirs(os.path.join(base_dir, pretrained_dir), exist_ok=True)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-b', '--base_dir', type=str, help='base directory', default='/datasets/models/taco2pt')
  
  args = parser.parse_args()

  ensure_folders_exist(args.base_dir)

  input_text_path = os.path.join(args.base_dir, input_text)
  input_text_exists = os.path.exists(input_text_path)

  if not input_text_exists:
    with open(input_text_path, 'w') as f:
      f.write('This is a test.')

  print('Initialized paths.')