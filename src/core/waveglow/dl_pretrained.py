import os
import shutil
import sys
import tempfile
import zipfile

import gdown
import wget

from src.core.common.utils import create_parent_folder
from src.core.waveglow.converter.convert import convert_main


def dl_wg_v3_and_convert(destination: str, auto_convert: bool, keep_orig: bool):
  if not os.path.exists(destination):
    print("Downloading pretrained waveglow model from Nvida...")
    # dl_v2(destination)
    dl_v3(destination)

  if auto_convert:
    print("Pretrained model is now beeing converted to be able to use it...")
    convert_glow(destination, destination, keep_orig)

# new file: wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/waveglow_ljs_256channels/versions/3/zip -O waveglow_ljs_256channels_3.zip
# https://ngc.nvidia.com/catalog/models/nvidia:waveglow_ljs_256channels


def dl_v3(destination: str):
  download_url = "https://api.ngc.nvidia.com/v2/models/nvidia/waveglow_ljs_256channels/versions/3/files/waveglow_256channels_ljs_v3.pt"
  create_parent_folder(destination)
  wget.download(download_url, destination)
  return
  # Download waveglow_universal_256channels_v5.pt (644M)
  download_url = "https://api.ngc.nvidia.com/v2/models/nvidia/waveglow_ljs_256channels/versions/3/zip"
  tmp_dl = tempfile.mktemp()
  tmp_dl = "/tmp/wg_v3.zip"
  #wget.download(download_url, tmp_dl)
  print(f"\nFinished download to {tmp_dl}")

  print("Unzipping...")
  parent_folder = create_parent_folder(destination)
  with zipfile.ZipFile(tmp_dl, 'r') as zip_ref:
    zip_ref.extractall(parent_folder)
  extracted_filepath = os.path.join(parent_folder, "waveglow_256channels_ljs_v3.pt")
  os.rename(extracted_filepath, destination)
  # os.remove(tmp_dl)


def dl_v2(destination: str):
  # Download waveglow_universal_256channels_v5.pt (644M)
  download_url = "https://drive.google.com/uc?id=1rpK8CzAAirq9sWZhe9nlfvxMF1dRgFbF"
  create_parent_folder(destination)
  gdown.download(download_url, destination)
  #shutil.move(filename, destination)


def convert_glow(origin: str, destination: str, keep_orig: bool = False):
  sys.path.append("src/core/waveglow/converter/")
  tmp_out = tempfile.mktemp()
  convert_main(origin, tmp_out)
  if keep_orig:
    if origin == destination:
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

  dl_v3(destination='/datasets/tmp/wg_v3/2.pt')

  # dl(
  #   destination='/tmp/testdl.pt',
  # )
  # convert_glow(
  #   #destination = '/datasets/models/pretrained/waveglow_256channels_universal_v5.pt',
  #   origin='/datasets/tmp/wg_v3/1.pt',
  #   destination='/datasets/tmp/wg_v3/1_conv.pt',
  #   keep_orig=True
  # )
