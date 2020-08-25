from argparse import ArgumentParser
from src.app import taco_train, taco_continue_train, taco_infer, taco_validate, taco_eval_checkpoints, taco_plot_embeddings
from src.core.common import Language
from src.cli.utils import parse_tuple_list
from typing import Optional, Tuple, List

def init_plot_emb_parser(parser):
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--train_name', type=str, required=True)
  parser.add_argument('--custom_checkpoint', type=int, default=0)
  return taco_plot_embeddings

def init_eval_checkpoints_parser(parser):
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--train_name', type=str, required=True)
  parser.add_argument('--custom_hparams', type=str)
  parser.add_argument('--select', type=int)
  parser.add_argument('--min_it', type=int)
  parser.add_argument('--max_it', type=int)
  return taco_eval_checkpoints

def init_train_parser(parser: ArgumentParser):
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--train_name', type=str, required=True)
  parser.add_argument('--prep_name', type=str, required=True)
  parser.add_argument('--warm_start_model', type=str)
  parser.add_argument('--test_size', type=float, default=0.001)
  parser.add_argument('--validation_size', type=float, default=0.1)
  parser.add_argument('--split_seed', type=int, default=1234)
  parser.add_argument('--hparams', type=str)
  parser.add_argument('--symbols_map_path', type=str)
  parser.add_argument('--emb_map_model', type=str)
  parser.add_argument('--emb_map_model_symbols', type=str)
  return taco_train

def init_continue_train_parser(parser: ArgumentParser):
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--train_name', type=str, required=True)
  parser.add_argument('--hparams', type=str)
  return taco_continue_train

def init_validate_parser(parser: ArgumentParser):
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--train_name', type=str, required=True)
  parser.add_argument('--entry_id', type=int, help="Utterance id or nothing if random")
  parser.add_argument('--ds_speaker', type=str, help="ds_name,speaker_name")
  parser.add_argument('--ds', type=str, help="Choose if validation- or testset should be taken.", choices=["val", "test"], default="val")
  parser.add_argument('--waveglow', type=str, help="Waveglow train_name", default="pretrained")
  parser.add_argument('--custom_checkpoint', type=int)
  parser.add_argument("--denoiser_strength", default=0.01, type=float, help='Removes model bias.')
  parser.add_argument("--sigma", default=0.666, type=float)
  parser.add_argument('--sampling_rate', type=float, default=22050)
  return taco_validate

def init_inference_parser(parser: ArgumentParser):
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--train_name', type=str, required=True)
  parser.add_argument('--ipa', action='store_true')
  parser.add_argument('--text', type=str, required=True)
  parser.add_argument('--lang', choices=Language, type=Language.__getitem__, required=True)
  parser.add_argument('--ignore_tones', action='store_true')
  #parser.add_argument('--ignore_arcs', action='store_true')
  parser.add_argument('--symbols_map', type=str, default="")
  parser.add_argument('--ds_speaker', type=str, help="ds_name,speaker_name", required=True)
  #parser.add_argument('--subset_id', type=str)
  parser.add_argument('--waveglow', type=str, help="Waveglow train_name", default="pretrained")
  parser.add_argument('--custom_checkpoint', type=int)
  parser.add_argument('--sentence_pause_s', type=float, default=0.5)
  parser.add_argument('--sigma', type=float, default=0.666)
  parser.add_argument('--denoiser_strength', type=float, default=0.01)
  parser.add_argument('--sampling_rate', type=float, default=22050)
  parser.add_argument('--analysis', action='store_true')
  parser.set_defaults(ignore_arcs=True)
  return taco_infer
