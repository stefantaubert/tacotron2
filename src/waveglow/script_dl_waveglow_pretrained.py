import argparse
import os
import gdown
import shutil
from pathlib import Path
from src.waveglow.converter.script_convert import convert

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--destination', type=str)
  parser.add_argument('--auto_convert', action='store_true')
  parser.add_argument('--no_debugging', action='store_true')
  
  args = parser.parse_args()
  
  if not args.no_debugging:
    args.destination = '/datasets/models/pretrained/waveglow_256channels_universal_v5.pt'
    args.destination = '/tmp/pretrained/waveglow_256channels_universal_v5.pt'
    args.auto_convert = True

  if not os.path.exists(args.destination):
    print("Downloading pretrained waveglow model from Nvida...")
    # Download waveglow_universal_256channels_v5.pt (644M)
    download_url = "https://drive.google.com/uc?id=1rpK8CzAAirq9sWZhe9nlfvxMF1dRgFbF"
    filename = gdown.download(download_url)
    path = Path(args.destination)
    os.makedirs(path.parent, exist_ok=True)
    shutil.move(filename, args.destination)

  if args.auto_convert:
    original_path = "{}.orig".format(args.destination)
    already_converted = os.path.exists(original_path)
    if already_converted:
      print("Pretrained model is already converted.")
    else:
      import sys
      sys.path.append("converter/")
      import tempfile
      tmp_out = tempfile.mktemp()
      convert(args.destination, tmp_out)
      shutil.move(args.destination, original_path)
      shutil.move(tmp_out, args.destination)