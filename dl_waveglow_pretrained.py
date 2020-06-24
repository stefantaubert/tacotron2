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

  download_url = "https://drive.google.com/uc?id=1rpK8CzAAirq9sWZhe9nlfvxMF1dRgFbF"
  res = gdown.download(download_url)
  
  shutil.move(res, os.path.join(args.pretrained_dir, res))

  print(res)
    