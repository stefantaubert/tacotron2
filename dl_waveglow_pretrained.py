import argparse
import os
import gdown
import shutil

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--pretrained_dir', type=str, help='base directory')
  parser.add_argument('--debug', type=str, default="true")
  
  args = parser.parse_args()
  debug = str.lower(args.debug) == 'true'
  if debug:
    args.pretrained_dir = '/tmp'

  # Download waveglow_universal_256channels_v5.pt (644M)
  download_url = "https://drive.google.com/uc?id=1rpK8CzAAirq9sWZhe9nlfvxMF1dRgFbF"
  filename = gdown.download(download_url)

  os.makedirs(args.pretrained_dir)
  shutil.move(filename, os.path.join(args.pretrained_dir, filename))
