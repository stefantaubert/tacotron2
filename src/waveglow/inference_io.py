import os
from shutil import copyfile
from src.core.common import args_to_str, parse_json
import datetime
from src.core.common import get_subdir

inference_dir = "inference"

def get_inference_dir(training_dir_path: str, input_name: str, checkpoint: str, speaker: str, create: bool = True) -> str:
  subdir_name = "{}_{}_{}_{}".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), input_name, speaker, checkpoint)
  return get_subdir(training_dir_path, os.path.join(inference_dir, subdir_name), create)
