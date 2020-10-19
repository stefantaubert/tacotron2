import logging
import os

import imageio

from src.app.io import get_train_root_dir
from src.core.common.mel_plot import compare_mels
from src.core.common.utils import (get_parent_dirname, get_subdir,
                                   stack_images_vertically)


def get_train_dir(base_dir: str, train_name: str, create: bool):
  return get_subdir(get_train_root_dir(base_dir, "waveglow", create), train_name, create)


def save_diff_plot(infer_dir: str):
  path1 = os.path.join(infer_dir, f"{get_parent_dirname(infer_dir)}.png")
  path2 = os.path.join(infer_dir, f"{get_parent_dirname(infer_dir)}_orig.png")

  old_level = logging.getLogger().level
  logging.getLogger().setLevel(logging.INFO)
  score, diff_img = compare_mels(path1, path2)
  path = os.path.join(infer_dir, f"{get_parent_dirname(infer_dir)}_diff.png")
  imageio.imsave(path, diff_img)
  logging.getLogger().setLevel(old_level)
  return score


def save_v(infer_dir: str):
  path1 = os.path.join(infer_dir, f"{get_parent_dirname(infer_dir)}_orig.png")
  path2 = os.path.join(infer_dir, f"{get_parent_dirname(infer_dir)}.png")
  path3 = os.path.join(infer_dir, f"{get_parent_dirname(infer_dir)}_diff.png")
  path = os.path.join(infer_dir, f"{get_parent_dirname(infer_dir)}_v.png")
  stack_images_vertically([path1, path2, path3], path)
