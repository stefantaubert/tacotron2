import os
import argparse

filelist_dir = "filelist"
input_dir = "in"
output_dir = "out"
log_dir = 'logs'
checkpoint_output_dir = 'output'
savecheckpoints_dir = 'saved_checkpoints'

symbols_path = os.path.join(filelist_dir, 'symbols.json')
symbols_path_info = os.path.join(filelist_dir, 'symbols.txt')

training_file = os.path.join(filelist_dir, 'audio_text_train_filelist.csv')
test_file = os.path.join(filelist_dir, 'audio_text_test_filelist.csv')
validation_file = os.path.join(filelist_dir, 'audio_text_val_filelist.csv')
preprocessed_file = os.path.join(filelist_dir, 'filelist.csv')
preprocessed_file_debug = os.path.join(filelist_dir, 'filelist_debug.csv')

input_symbols = os.path.join(input_dir, 'input_symbols.txt')

def ensure_folders_exist(base_dir):
  os.makedirs(os.path.join(base_dir, filelist_dir), exist_ok=True)
  os.makedirs(os.path.join(base_dir, input_dir), exist_ok=True)
  os.makedirs(os.path.join(base_dir, output_dir), exist_ok=True)
  os.makedirs(os.path.join(base_dir, log_dir), exist_ok=True)
  os.makedirs(os.path.join(base_dir, checkpoint_output_dir), exist_ok=True)
  os.makedirs(os.path.join(base_dir, savecheckpoints_dir), exist_ok=True)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-b', '--base_dir', type=str, help='base directory', default='/datasets/models/taco2pt_testing')
  
  args = parser.parse_args()

  ensure_folders_exist(args.base_dir)

  # input_text_path = os.path.join(args.base_dir, input_text)
  # input_text_exists = os.path.exists(input_text_path)

  # if not input_text_exists:
  #   with open(input_text_path, 'w') as f:
  #     f.write('This is a test.')

  print('Initialized paths for {}.'.format(args.base_dir))