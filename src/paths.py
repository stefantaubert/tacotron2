import os
import datetime
from shutil import copyfile
from src.common.utils import args_to_str

log_dir = 'logs'
log_train_file_name = 'log.txt'

analysis_dir = 'analysis'
analysis_sims_file_name = 'similarities.txt'
analysis_2d_file_name = '2d.html'
analysis_3d_file_name = '3d.html'

filelist_dir = "filelist"
filelist_training_file_name = 'audio_text_train_filelist.csv'
filelist_test_file_name = 'audio_text_test_filelist.csv'
filelist_validation_file_name = 'audio_text_val_filelist.csv'
filelist_symbols_file_name = 'symbols.json'
filelist_file_name = 'filelist.csv'
filelist_file_log_name = 'filelist_log.csv'
filelist_speakers_name = 'speakers.json'
filelist_weights_file_name = 'weights.npy'

ds_dir = 'ds'
ds_preprocessed_file_name = 'filelist.csv'
ds_preprocessed_symbols_name = 'symbols.json'
ds_preprocessed_all_symbols_name = 'all_symbols.json'
ds_preprocessed_all_speakers_name = 'all_speakers.json'

mels_dir = 'mels'
mels_file_name = 'mels.csv'

wavs_dir = 'wavs'
wavs_file_name = 'wavs.csv'

inference_dir = 'inference'
inference_input_file_name = 'input.txt'
inference_input_map_file_name = 'input_map.json'
inference_input_normalized_sentences_file_name = 'input_normalized_sentences.txt'
inference_input_sentences_file_name = 'input_sentences.txt'
inference_input_sentences_mapped_file_name = 'input_sentences_mapped.txt'
inference_input_symbols_file_name = 'input_symbols.txt'
inference_config_file = 'config.log'

checkpoint_dir = 'checkpoints'

train_config_file = 'config.log'
train_map_file = 'weights_map.json'
description_txt_file = 'description.txt'

def get_training_dir(base_dir: str, create: bool = True) -> str:
  training_dir = "training_{}".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
  training_dir_path = os.path.join(base_dir, training_dir)
  if create:
    os.makedirs(training_dir_path, exist_ok=True)
  return training_dir

def __get_subdir(training_dir_path: str, subdir: str, create: bool = True) -> str:
  result = os.path.join(training_dir_path, subdir)
  if create:
    os.makedirs(result, exist_ok=True)
  return result

def get_symbols_path(training_dir_path: str) -> str:
  path = os.path.join(get_filelist_dir(training_dir_path), filelist_symbols_file_name)
  return path

def get_all_symbols_path(base_dir: str, name: str, create: bool = True) -> str:
  path = os.path.join(__get_subdir(base_dir, os.path.join(ds_dir, name), create), ds_preprocessed_all_symbols_name)
  return path

def get_all_speakers_path(base_dir: str, name: str, create: bool = True) -> str:
  path = os.path.join(__get_subdir(base_dir, os.path.join(ds_dir, name), create), ds_preprocessed_all_speakers_name)
  return path

def get_ds_dir(base_dir: str, name: str, speaker: str, create: bool = True) -> str:
  return __get_subdir(base_dir, os.path.join(ds_dir, name, str(speaker)), create)

def get_wavs_dir(base_dir: str, name: str, create: bool = True) -> str:
  return __get_subdir(base_dir, os.path.join(wavs_dir, name), create)

def get_mels_dir(base_dir: str, name: str, create: bool = True) -> str:
  return __get_subdir(base_dir, os.path.join(mels_dir, name), create)

def get_analysis_dir(training_dir_path: str, create: bool = True) -> str:
  return __get_subdir(training_dir_path, analysis_dir, create)

def get_filelist_dir(training_dir_path: str, create: bool = True) -> str:
  return __get_subdir(training_dir_path, filelist_dir, create)

def get_checkpoint_dir(training_dir_path: str, create: bool = True) -> str:
  return __get_subdir(training_dir_path, checkpoint_dir, create)

def get_log_dir(training_dir_path: str, create: bool = True) -> str:
  return __get_subdir(training_dir_path, log_dir, create)

def get_inference_dir(training_dir_path: str, input_name: str, checkpoint: str, speaker: str, create: bool = True) -> str:
  subdir_name = "{}_{}_{}_{}".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), input_name, speaker, checkpoint)
  return __get_subdir(training_dir_path, os.path.join(inference_dir, subdir_name), create)

def get_validation_dir(training_dir_path: str, i, input_name: str, checkpoint: str, speaker: str, create: bool = True) -> str:
  subdir_name = "validation_{}_({})_{}_{}_{}".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), i, input_name, speaker, checkpoint)
  return __get_subdir(training_dir_path, os.path.join(inference_dir, subdir_name), create)

def log_train_config(training_dir_path: str, args):
  t = args_to_str(args)
  with open(os.path.join(training_dir_path, train_config_file), 'w', encoding='utf-8') as f:
    f.write(t)
  print("Passed training arguments:")
  print(t)

def log_train_map(training_dir_path: str, map_path: str):
  assert map_path
  copyfile(map_path, os.path.join(training_dir_path, train_map_file))

def log_inference_config(infer_dir_path: str, args):
  t = args_to_str(args)
  with open(os.path.join(infer_dir_path, inference_config_file), 'w', encoding='utf-8') as f:
    f.write(t)
  print("Passed inference arguments:")
  print(t)

def log_input_file(infer_dir_path: str, input_file: str):
  copyfile(input_file, os.path.join(infer_dir_path, inference_input_file_name))

def log_map_file(infer_dir_path: str, map_file: str):
  copyfile(map_file, os.path.join(infer_dir_path, inference_input_map_file_name))

def create_description_file(training_dir_path: str):
  desc_file = os.path.join(training_dir_path, description_txt_file)
  if not os.path.exists(desc_file):
    with open(desc_file, 'w', encoding='utf-8') as f:
      f.write('Description\n')
      f.write('-----------\n')

def init_path_parser(parser):
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--custom_training_name', type=str)
  return main

def main(base_dir: str, custom_training_name: str):
  print(base_dir)
  print(custom_training_name)
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

if __name__ == "__main__":
  main(
    base_dir='/datasets/models/taco2pt_v2',
    custom_training_name='debug_ljs_ms_test'
  )
