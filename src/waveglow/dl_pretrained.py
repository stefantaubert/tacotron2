import argparse
import os
import gdown
import shutil
from pathlib import Path
from src.waveglow.converter.convert import __convert
from src.common.utils import create_parent_folder

def init_download_parser(parser):
  parser.add_argument('--destination', type=str, required=True)
  parser.add_argument('--auto_convert', action='store_true')
  return main

def main(destination, auto_convert):
  if not os.path.exists(destination):
    print("Downloading pretrained waveglow model from Nvida...")
    # Download waveglow_universal_256channels_v5.pt (644M)
    download_url = "https://drive.google.com/uc?id=1rpK8CzAAirq9sWZhe9nlfvxMF1dRgFbF"
    filename = gdown.download(download_url)
    create_parent_folder(destination)
    shutil.move(filename, destination)

  if auto_convert:
    original_path = "{}.orig".format(destination)
    already_converted = os.path.exists(original_path)
    if already_converted:
      print("Pretrained model is already converted.")
    else:
      import sys
      sys.path.append("converter/")
      import tempfile
      tmp_out = tempfile.mktemp()
      __convert(destination, tmp_out)
      shutil.move(destination, original_path)
      shutil.move(tmp_out, destination)

if __name__ == "__main__":
  main(
    #destination = '/datasets/models/pretrained/waveglow_256channels_universal_v5.pt',
    destination = '/tmp/pretrained/waveglow_256channels_universal_v5.pt',
    auto_convert = True
  )