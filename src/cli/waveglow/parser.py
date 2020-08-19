from argparse import ArgumentParser
from src.cli.waveglow.main import train, continue_train, infer, validate


def init_train_parser(parser: ArgumentParser):
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--train_name', type=str, required=True)
  parser.add_argument('--fl_name', type=str, required=True)
  parser.add_argument('--test_size', type=float, default=0.001)
  parser.add_argument('--validation_size', type=float, default=0.1)
  parser.add_argument('--split_seed', type=int, default=1234)
  parser.add_argument('--hparams', type=str)
  return train

def init_continue_train_parser(parser: ArgumentParser):
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--train_name', type=str, required=True)
  parser.add_argument('--hparams', type=str)
  return continue_train

def init_validate_parser(parser):
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--train_name', type=str, required=True)
  parser.add_argument('--entry_id', type=int, help="Utterance id", required=True)
  parser.add_argument('--custom_checkpoint', type=int, default=0)
  parser.add_argument("--denoiser_strength", default=0.01, type=float, help='Removes model bias.')
  parser.add_argument("--sigma", default=0.666, type=float)
  parser.add_argument('--sampling_rate', type=float, default=22050)
  return validate

def init_inference_parser(parser):
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--train_name', type=str, required=True)
  parser.add_argument('--wav_path', type=str, required=True)
  parser.add_argument('--custom_checkpoint', type=str, default="")
  parser.add_argument('--sigma', type=float, default=0.666)
  parser.add_argument('--denoiser_strength', type=float, default=0.01)
  parser.add_argument('--sampling_rate', type=float, default=22050)
  return infer
