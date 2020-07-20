import argparse
import os
import gdown
import shutil

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--pretrained_dir', type=str, help='base directory')
  parser.add_argument('--no_debugging', action='store_true')
  
  args = parser.parse_args()
  
  if not args.no_debugging:
    args.pretrained_dir = '/datasets/models/pretrained/'

  dest_name = os.path.join(args.pretrained_dir, 'waveglow_256channels_universal_v5.pt')

  if not os.path.exists(dest_name):
    print("Downloading pretrained waveglow model from Nvida...")
    # Download waveglow_universal_256channels_v5.pt (644M)
    download_url = "https://drive.google.com/uc?id=1rpK8CzAAirq9sWZhe9nlfvxMF1dRgFbF"
    filename = gdown.download(download_url)

    os.makedirs(args.pretrained_dir)
    shutil.move(filename, dest_name)
