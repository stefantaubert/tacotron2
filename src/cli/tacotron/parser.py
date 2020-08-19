from argparse import ArgumentParser
from src.cli.tacotron.main import train, continue_train, infer, validate


def init_train_parser(parser: ArgumentParser):
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--train_name', type=str, required=True)
  parser.add_argument('--fl_name', type=str, required=True)
  parser.add_argument('--warm_start_model', type=str)
  parser.add_argument('--test_size', type=float, default=0.001)
  parser.add_argument('--validation_size', type=float, default=0.1)
  parser.add_argument('--split_seed', type=int, default=1234)
  parser.add_argument('--hparams', type=str)
  parser.add_argument('--weight_map_mode', type=str, choices=['same_symbols_only', 'use_map'])
  parser.add_argument('--weight_map', type=str)
  parser.add_argument('--weight_map_model', type=str)
  parser.add_argument('--weight_map_model_symbols', type=str)
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
  parser.add_argument('--waveglow', type=str, required=True)
  parser.add_argument('--custom_checkpoint', type=int, default=0)
  parser.add_argument("--denoiser_strength", default=0.01, type=float, help='Removes model bias.')
  parser.add_argument("--sigma", default=0.666, type=float)
  parser.add_argument('--sampling_rate', type=float, default=22050)
  return validate

def init_inference_parser(parser):
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--train_name', type=str, required=True)
  parser.add_argument('--ipa', action='store_true')
  parser.add_argument('--text', type=str, required=True)
  parser.add_argument('--lang', type=int, choices=[0, 1, 2, 3], required=True)
  parser.add_argument('--ignore_tones', action='store_true')
  parser.add_argument('--ignore_arcs', action='store_true')
  parser.add_argument('--symbols_map', type=str, default="")
  parser.add_argument('--speaker_id', type=int, required=True)
  #parser.add_argument('--subset_id', type=str)
  parser.add_argument('--waveglow', type=str, required=True)
  parser.add_argument('--custom_checkpoint', type=str, default="")
  parser.add_argument('--sentence_pause_s', type=float, default=0.5)
  parser.add_argument('--sigma', type=float, default=0.666)
  parser.add_argument('--denoiser_strength', type=float, default=0.01)
  parser.add_argument('--sampling_rate', type=float, default=22050)
  parser.add_argument('--analysis', action='store_true')
  return infer