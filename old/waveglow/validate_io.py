from src.core.common import get_subdir
import datetime
import os

validation_dir = "validation"
inference_input_file_name = "input.txt"

def get_validation_dir(training_dir_path: str, i, input_name: str, checkpoint: str, speaker: str, create: bool = True) -> str:
  subdir_name = "{}_id-{}_({})_{}_{}".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), i, input_name, speaker, checkpoint)
  return get_subdir(training_dir_path, os.path.join(validation_dir, subdir_name), create)

def write_input_file(infer_dir_path: str, orig_text):
  with open(os.path.join(infer_dir_path, inference_input_file_name), 'w', encoding='utf-8') as f:
    f.writelines([orig_text])