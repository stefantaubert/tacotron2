import os
from shutil import copyfile
from src.common.utils import args_to_str, parse_json
import datetime
from src.common.utils import get_subdir

inference_dir = 'inference'
inference_input_file_name = 'input.txt'
inference_input_map_file_name = 'input_map.json'
inference_input_normalized_sentences_file_name = 'input_normalized_sentences.txt'
inference_input_sentences_file_name = 'input_sentences.txt'
inference_input_sentences_mapped_file_name = 'input_sentences_mapped.txt'
inference_input_symbols_file_name = 'input_symbols.txt'
inference_config_file = 'config.log'

def log_input_file(infer_dir_path: str, input_file: str):
  copyfile(input_file, os.path.join(infer_dir_path, inference_input_file_name))

def log_map_file(infer_dir_path: str, map_file: str):
  copyfile(map_file, os.path.join(infer_dir_path, inference_input_map_file_name))

def get_inference_dir(training_dir_path: str, input_name: str, checkpoint: str, speaker: str, create: bool = True) -> str:
  subdir_name = "{}_{}_{}_{}".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), input_name, speaker, checkpoint)
  return get_subdir(training_dir_path, os.path.join(inference_dir, subdir_name), create)

def log_inference_config(infer_dir_path: str, args):
  t = args_to_str(args)
  with open(os.path.join(infer_dir_path, inference_config_file), 'w', encoding='utf-8') as f:
    f.write(t)
  print("Passed inference arguments:")
  print(t)

def parse_input_file(infer_dir_path: str):
  input_file = os.path.join(infer_dir_path, inference_input_file_name)
  with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()
  return lines

def write_input_normalized_sentences(infer_dir_path: str, cleaned_sents):
  with open(os.path.join(infer_dir_path, inference_input_normalized_sentences_file_name), 'w', encoding='utf-8') as f:
    f.writelines(['{}\n'.format(s) for s in cleaned_sents])
   
def write_input_sentences(infer_dir_path: str, accented_sents):
  with open(os.path.join(infer_dir_path, inference_input_sentences_file_name), 'w', encoding='utf-8') as f:
    f.writelines(['{}\n'.format(s) for s in accented_sents])

def write_input_symbols(infer_dir_path: str, seq_sents):
  with open(os.path.join(infer_dir_path, inference_input_symbols_file_name), 'w', encoding='utf-8') as f:
    f.writelines(seq_sents)
  
def write_input_sentences_mapped(infer_dir_path: str, seq_sents_text):
  with open(os.path.join(infer_dir_path, inference_input_sentences_mapped_file_name), 'w', encoding='utf-8') as f:
    f.writelines(['{}\n'.format(s) for s in seq_sents_text])

def parse_map(infer_dir_path: str):
  map_path = os.path.join(infer_dir_path, inference_input_map_file_name)
  ipa_mapping = parse_json(map_path)
  return ipa_mapping