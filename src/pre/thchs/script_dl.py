import argparse

from src.parser.thchs_parser import ensure_downloaded
from src.parser.thchs_kaldi_parser import ensure_downloaded as kaldi_ensure_downloaded

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--no_debugging', action='store_true')
  parser.add_argument('--kaldi_version', action='store_true')
  parser.add_argument('--data_dir', type=str, help='THCHS dataset directory')

  args = parser.parse_args()
  
  if not args.no_debugging:
    args.kaldi_version = True
    args.kaldi_version = False
    if args.kaldi_version:
      args.data_dir = '/datasets/THCHS-30'
    else:
      args.data_dir = '/datasets/thchs_wav'
  
  if args.kaldi_version:
    kaldi_ensure_downloaded(args.data_dir)
  else:
    ensure_downloaded(args.data_dir)
