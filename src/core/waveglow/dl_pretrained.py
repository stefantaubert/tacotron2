import os
import shutil
import sys
import tempfile

import gdown

from src.core.common.utils import create_parent_folder
from src.core.waveglow.converter import convert


def main(destination: str, auto_convert: bool = True, keep_orig: bool = False):
  if not os.path.exists(destination):
    print("Downloading pretrained waveglow model from Nvida...")
    dl(destination)

  if auto_convert:
    print("Pretrained model is now beeing converted to be able to use it...")
    convert_glow(destination, destination, keep_orig)

# new file: wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/waveglow_ljs_256channels/versions/3/zip -O waveglow_ljs_256channels_3.zip
# https://ngc.nvidia.com/catalog/models/nvidia:waveglow_ljs_256channels

def dl(destination: str):
  # Download waveglow_universal_256channels_v5.pt (644M)
  download_url = "https://drive.google.com/uc?id=1rpK8CzAAirq9sWZhe9nlfvxMF1dRgFbF"
  create_parent_folder(destination)
  gdown.download(download_url, destination)
  #shutil.move(filename, destination)

def convert_glow(origin: str, destination: str, keep_orig: bool = False):
  sys.path.append("src/core/waveglow/converter/")
  tmp_out = tempfile.mktemp()
  convert(origin, tmp_out)
  if keep_orig:
    original_path = "{}.orig".format(origin)
    shutil.move(origin, original_path)
  else:
    os.remove(origin)
  shutil.move(tmp_out, destination)

if __name__ == "__main__":
  # main(
  #   #destination = '/datasets/models/pretrained/waveglow_256channels_universal_v5.pt',
  #   destination = '/tmp/testdl.pt',
  #   auto_convert = True,
  #   keep_orig = False
  # )
  dl(
    destination = '/tmp/testdl.pt',
  )
  convert_glow(
    #destination = '/datasets/models/pretrained/waveglow_256channels_universal_v5.pt',
    origin = '/tmp/testdl.pt',
    destination = '/tmp/testdl_conv.pt',
    keep_orig = True
  )
