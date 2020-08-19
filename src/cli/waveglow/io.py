import datetime
import os
from shutil import copyfile
from typing import List

import matplotlib.pylab as plt
import numpy as np
import imageio
from tqdm import tqdm

from src.core.common import (Language, float_to_wav, get_basename, compare_mels,
                             get_custom_or_last_checkpoint, get_parent_dirname,
                             get_subdir, parse_json, plot_melspec,
                             stack_images_vertically)
from src.core.pre import PreparedDataList, SpeakersIdDict, SymbolConverter

from src.cli.io import *

def get_infer_dir(base_dir: str, train_name: str, input_name: str, iteration: int):
  subdir_name = "{}_wav-{}_it-{}".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), input_name, iteration)
  return get_subdir(get_inference_root_dir(base_dir, train_name), subdir_name, create=True)

def save_infer_orig_plot(infer_dir: str, mel: np.ndarray):
  plot_melspec(mel, title="Original")
  path = os.path.join(infer_dir, f"{get_parent_dirname(infer_dir)}_orig.png")
  plt.savefig(path, bbox_inches='tight')
  return path

def save_infer_orig_wav(infer_dir: str, wav_path_orig: str):
  path = os.path.join(infer_dir, f"{get_parent_dirname(infer_dir)}_orig.wav")
  copyfile(wav_path_orig, path)

def save_diff_plot(infer_dir: str):
  path1 = os.path.join(infer_dir, f"{get_parent_dirname(infer_dir)}.png")
  path2 = os.path.join(infer_dir, f"{get_parent_dirname(infer_dir)}_orig.png")
  _, diff_img = compare_mels(path1, path2)
  path = os.path.join(infer_dir, f"{get_parent_dirname(infer_dir)}_diff.png")
  imageio.imsave(path, diff_img)

def save_v(infer_dir: str):
  path1 = os.path.join(infer_dir, f"{get_parent_dirname(infer_dir)}.png")
  path2 = os.path.join(infer_dir, f"{get_parent_dirname(infer_dir)}_orig.png")
  path3 = os.path.join(infer_dir, f"{get_parent_dirname(infer_dir)}_diff.png")
  path = os.path.join(infer_dir, f"{get_parent_dirname(infer_dir)}_v.png")
  stack_images_vertically([path1, path2, path3], path)
