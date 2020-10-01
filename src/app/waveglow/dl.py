import os
from src.app.tacotron.defaults import DEFAULT_WAVEGLOW
from typing import Optional

from src.app.io import (get_checkpoints_dir, save_prep_name, save_testset,
                        save_valset)
from src.app.pre.prepare import get_prepared_dir, load_filelist
from src.app.waveglow.io import get_train_dir
from src.core.common.train import get_pytorch_filename
from src.core.waveglow.converter.convert import convert_glow
from src.core.waveglow.dl_pretrained import dl_wg


def get_checkpoint_pretrained(checkpoints_dir: str):
  return os.path.join(checkpoints_dir, get_pytorch_filename("1"))


def dl_pretrained(base_dir: str, train_name: str = DEFAULT_WAVEGLOW, prep_name: Optional[str] = None, version: int = 3):
  train_dir = get_train_dir(base_dir, train_name, create=True)
  assert os.path.isdir(train_dir)
  checkpoints_dir = get_checkpoints_dir(train_dir)
  dest_path = get_checkpoint_pretrained(checkpoints_dir)

  print("Downloading pretrained waveglow model from Nvida...")
  dl_wg(
    destination=dest_path,
    version=version
  )

  print("Pretrained model is now beeing converted to be able to use it...")
  convert_glow(
    origin=dest_path,
    destination=dest_path,
    keep_orig=False
  )

  if prep_name is not None:
    prep_dir = get_prepared_dir(base_dir, prep_name)
    wholeset = load_filelist(prep_dir)
    save_testset(train_dir, wholeset)
    save_valset(train_dir, wholeset)
  save_prep_name(train_dir, prep_name=prep_name)


if __name__ == "__main__":
  dl_pretrained(
    version=3,
    train_name="pretrained_v3",
    base_dir="/datasets/models/taco2pt_v5",
    prep_name="ljs_ipa",
  )
