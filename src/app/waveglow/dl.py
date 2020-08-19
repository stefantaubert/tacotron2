import os

from src.app.io import get_checkpoints_dir
from src.app.waveglow.io import get_train_dir
from src.core.common import get_pytorch_filename
from src.core.waveglow import dl_pretrained as dl_pretrained_core


def get_checkpoint_pretrained(checkpoints_dir: str):
  return os.path.join(checkpoints_dir, get_pytorch_filename("1"))

def dl_pretrained(base_dir: str, train_name: str = "pretrained"):
  train_dir = get_train_dir(base_dir, train_name, create=True)
  assert os.path.isdir(train_dir)
  checkpoints_dir = get_checkpoints_dir(train_dir)
  dest_path = get_checkpoint_pretrained(checkpoints_dir)
  dl_pretrained_core(dest_path, auto_convert=True, keep_orig=False)

if __name__ == "__main__":
  dl_pretrained(
    base_dir="/datasets/models/taco2pt_v3",
  )
