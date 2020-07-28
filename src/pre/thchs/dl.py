import argparse

from src.parser.thchs_parser import ensure_downloaded
from src.parser.thchs_kaldi_parser import ensure_downloaded as kaldi_ensure_downloaded

def main(kaldi_version: bool, data_dir: str):
  if kaldi_version:
    kaldi_ensure_downloaded(data_dir)
  else:
    ensure_downloaded(data_dir)


if __name__ == "__main__":
  kaldi_version = True
  kaldi_version = False
  if kaldi_version:
    data_dir = '/datasets/THCHS-30'
  else:
    data_dir = '/datasets/thchs_wav'
  