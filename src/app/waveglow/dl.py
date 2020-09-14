import os
from typing import Optional

from src.app.io import (get_checkpoints_dir, save_settings, save_testset,
                        save_valset)
from src.app.pre.prepare import get_prepared_dir, load_filelist
from src.app.waveglow.io import get_train_dir
from src.core.common.train import get_pytorch_filename
from src.core.waveglow.dl_pretrained import dl_wg_v3_and_convert


def get_checkpoint_pretrained(checkpoints_dir: str):
  return os.path.join(checkpoints_dir, get_pytorch_filename("1"))


def dl_pretrained(base_dir: str, train_name: str = "pretrained", prep_name: Optional[str] = None):
  train_dir = get_train_dir(base_dir, train_name, create=True)
  assert os.path.isdir(train_dir)
  checkpoints_dir = get_checkpoints_dir(train_dir)
  dest_path = get_checkpoint_pretrained(checkpoints_dir)
  dl_wg_v3_and_convert(dest_path, auto_convert=True, keep_orig=False)
  if prep_name is not None:
    prep_dir = get_prepared_dir(base_dir, prep_name)
    wholeset = load_filelist(prep_dir)
    save_testset(train_dir, wholeset)
    save_valset(train_dir, wholeset)
  save_settings(train_dir, prep_name=prep_name, custom_hparams=None)


if __name__ == "__main__":
  dl_pretrained(
    base_dir="/datasets/models/taco2pt_v5",
    train_name="pretrained",
    prep_name="ljs_ipa"
  )
