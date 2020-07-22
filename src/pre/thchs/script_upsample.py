import argparse
from src.pre.thchs.upsample import ensure_upsampled


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_src_dir', type=str, help='THCHS dataset directory')
  parser.add_argument('--data_dest_dir', type=str, help='THCHS destination directory')
  parser.add_argument('--kaldi_version', action='store_true')
  parser.add_argument('--no_debugging', action='store_true')

  args = parser.parse_args()

  if not args.no_debugging:
    args.data_src_dir = '/datasets/thchs_wav'
    args.data_dest_dir = '/datasets/thchs_16bit_22050kHz'
    args.kaldi_version = False

  ensure_upsampled(args.data_src_dir, args.data_dest_dir, args.kaldi_version)
