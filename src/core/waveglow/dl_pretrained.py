import os

import gdown
import wget

from src.core.common.utils import create_parent_folder

# src: https://ngc.nvidia.com/catalog/models/nvidia:waveglow_ljs_256channels

def dl_wg(destination: str, version: int):
  if version == 1:
    _dl_v1(destination)
  elif version == 2:
    _dl_v2(destination)
  elif version == 3:
    _dl_v3(destination)
  else:
    assert False


def _dl_v3(destination: str):
  download_url = "https://api.ngc.nvidia.com/v2/models/nvidia/waveglow_ljs_256channels/versions/3/files/waveglow_256channels_ljs_v3.pt"
  create_parent_folder(destination)
  wget.download(download_url, destination)


def _dl_v2(destination: str):
  download_url = "https://api.ngc.nvidia.com/v2/models/nvidia/waveglow_ljs_256channels/versions/2/files/waveglow_256channels_ljs_v2.pt"
  create_parent_folder(destination)
  wget.download(download_url, destination)


def _dl_v1(destination: str):
  # Download waveglow_universal_256channels_v5.pt (644M)
  download_url = "https://drive.google.com/uc?id=1rpK8CzAAirq9sWZhe9nlfvxMF1dRgFbF"
  create_parent_folder(destination)
  gdown.download(download_url, destination)


if __name__ == "__main__":
  # dl_wg(
  #   #destination = '/datasets/models/pretrained/waveglow_256channels_universal_v5.pt',
  #   destination = '/tmp/testdl.pt',
  # )

  # _dl_v1(destination='/datasets/tmp/wg_v3/v1.pt')
  # _dl_v2(destination='/datasets/tmp/wg_v3/v2.pt')
  _dl_v3(destination='/datasets/tmp/wg_v3/v3.pt')
