import argparse
from src.pre.thchs.remove_silence import remove_silence_thchs
from src.parser.thchs_parser import parse, exists
from src.parser.thchs_kaldi_parser import parse as kaldi_parse, exists as kaldi_exists

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_src_dir', type=str, help='THCHS dataset directory')
  parser.add_argument('--data_dest_dir', type=str, help='THCHS destination directory')
  parser.add_argument('--kaldi_version', action='store_true')
  parser.add_argument('--no_debugging', action='store_true')
  parser.add_argument('--chunk_size', type=int)
  parser.add_argument('--threshold_start', type=float)
  parser.add_argument('--threshold_end', type=float)
  parser.add_argument('--buffer_start_ms', type=float, help="amount of factors of chunk_size at the beginning and the end should be reserved")
  parser.add_argument('--buffer_end_ms', type=float, help="amount of factors of chunk_size at the beginning and the end should be reserved")

  args = parser.parse_args()

  if not args.no_debugging:
    args.data_src_dir = '/datasets/thchs_16bit_22050kHz'
    args.data_dest_dir = '/datasets/thchs_16bit_22050kHz_nosil'
    args.kaldi_version = False
    args.chunk_size = 5
    args.threshold_start = -25
    args.threshold_end = -35
    args.buffer_start_ms = 100
    args.buffer_end_ms = 150

  if args.kaldi_version:
    already_removed = kaldi_exists(args.data_dest_dir)
  else:
    already_removed = exists(args.data_dest_dir)
  
  if already_removed:
    print("Dataset is already without silence.")
  else:
    print("Saving to {}".format(args.data_dest_dir))
    
    remove_silence_thchs(
      origin=args.data_src_dir,
      dest=args.data_dest_dir,
      kaldi_version=args.kaldi_version,
      chunk_size = args.chunk_size,
      threshold_start = args.threshold_start,
      threshold_end = args.threshold_end,
      buffer_start_ms = args.buffer_start_ms,
      buffer_end_ms = args.buffer_end_ms
    )

    print("Finished.")
