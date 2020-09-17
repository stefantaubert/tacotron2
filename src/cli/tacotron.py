from argparse import ArgumentParser

from src.app.tacotron.analysis import plot_embeddings
from src.app.tacotron.defaults import (DEFAULT_DENOISER_STRENGTH,
                                       DEFAULT_SAMPLING_RATE,
                                       DEFAULT_SENTENCE_PAUSE_S, DEFAULT_SIGMA,
                                       DEFAULT_WAVEGLOW)
from src.app.tacotron.eval_checkpoints import eval_checkpoints_main
from src.app.tacotron.inference import infer
from src.app.tacotron.training import continue_train, train
from src.app.tacotron.validation import validate


def init_plot_emb_parser(parser):
  parser.add_argument('--train_name', type=str, required=True)
  parser.add_argument('--custom_checkpoint', type=int, default=0)
  return plot_embeddings


def init_eval_checkpoints_parser(parser):
  parser.add_argument('--train_name', type=str, required=True)
  parser.add_argument('--custom_hparams', type=str)
  parser.add_argument('--select', type=int)
  parser.add_argument('--min_it', type=int)
  parser.add_argument('--max_it', type=int)
  return eval_checkpoints_main


def init_train_parser(parser: ArgumentParser):
  parser.add_argument('--train_name', type=str, required=True)
  parser.add_argument('--prep_name', type=str, required=True)
  parser.add_argument('--test_size', type=float, default=0.001)
  parser.add_argument('--validation_size', type=float, default=0.1)
  parser.add_argument('--split_seed', type=int, default=1234)
  parser.add_argument('--custom_hparams', type=str)
  parser.add_argument('--warm_start_train_name', type=str)
  parser.add_argument('--warm_start_checkpoint', type=int)
  parser.add_argument('--weights_train_name', type=str)
  parser.add_argument('--weights_checkpoint', type=int)
  parser.add_argument('--weights_map', type=str)
  return train


def init_continue_train_parser(parser: ArgumentParser):
  parser.add_argument('--train_name', type=str, required=True)
  parser.add_argument('--custom_hparams', type=str)
  return continue_train


def init_validate_parser(parser: ArgumentParser):
  parser.add_argument('--train_name', type=str, required=True)
  parser.add_argument('--entry_id', type=int, help="Utterance id or nothing if random")
  parser.add_argument('--ds_speaker', type=str, help="ds_name,speaker_name")
  parser.add_argument('--ds', type=str, help="Choose if validation- or testset should be taken.",
                      choices=["val", "test"], default="val")
  parser.add_argument('--waveglow', type=str, help="Waveglow train_name", default=DEFAULT_WAVEGLOW)
  parser.add_argument('--custom_checkpoint', type=int)
  parser.add_argument("--denoiser_strength", default=DEFAULT_DENOISER_STRENGTH,
                      type=float, help='Removes model bias.')
  parser.add_argument("--sigma", default=DEFAULT_SIGMA, type=float)
  return validate


def init_inference_parser(parser: ArgumentParser):
  parser.add_argument('--train_name', type=str, required=True)
  parser.add_argument('--text_name', type=str, required=True)
  parser.add_argument('--ds_speaker', type=str, help="ds_name,speaker_name", required=True)
  parser.add_argument('--waveglow', type=str, help="Waveglow train_name", default=DEFAULT_WAVEGLOW)
  parser.add_argument('--custom_checkpoint', type=int)
  parser.add_argument('--sentence_pause_s', type=float, default=DEFAULT_SENTENCE_PAUSE_S)
  parser.add_argument(
    '--sigma',
    type=float,
    default=DEFAULT_SIGMA
  )
  parser.add_argument(
    '--denoiser_strength',
    type=float,
    default=DEFAULT_DENOISER_STRENGTH
  )
  parser.add_argument('--analysis', action='store_true')
  return infer
