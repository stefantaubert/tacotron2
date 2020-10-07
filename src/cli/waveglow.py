from argparse import ArgumentParser

from src.app.tacotron.defaults import DEFAULT_WAVEGLOW
from src.app.waveglow.dl import dl_pretrained
from src.app.waveglow.inference import infer
from src.app.waveglow.training import continue_training, start_new_training
from src.app.waveglow.validation import validate
from src.cli.utils import split_hparams_string


def init_train_parser(parser: ArgumentParser):
  parser.add_argument('--train_name', type=str, required=True)
  parser.add_argument('--prep_name', type=str, required=True)
  parser.add_argument('--test_size', type=float, default=0.001)
  parser.add_argument('--validation_size', type=float, default=0.1)
  parser.add_argument('--split_seed', type=int, default=1234)
  parser.add_argument('--custom_hparams', type=str)
  parser.add_argument('--warm_start_train_name', type=str)
  parser.add_argument('--warm_start_checkpoint', type=int)
  return train_cli


def train_cli(**args):
  args["custom_hparams"] = split_hparams_string(args["custom_hparams"])
  start_new_training(**args)


def init_continue_train_parser(parser: ArgumentParser):
  parser.add_argument('--train_name', type=str, required=True)
  parser.add_argument('--custom_hparams', type=str)
  return continue_train_cli


def continue_train_cli(**args):
  args["custom_hparams"] = split_hparams_string(args["custom_hparams"])
  continue_training(**args)


def init_validate_parser(parser: ArgumentParser):
  parser.add_argument('--train_name', type=str, required=True)
  parser.add_argument('--entry_id', type=int, help="Utterance id or nothing if random")
  parser.add_argument('--ds_speaker', type=str, help="ds_name,speaker_name")
  parser.add_argument('--ds', type=str, help="Choose if validation- or testset should be taken.",
                      choices=["val", "test"], default="val")
  parser.add_argument('--custom_checkpoint', type=int)
  parser.add_argument("--denoiser_strength", default=0.00, type=float, help='Removes model bias.')
  parser.add_argument("--sigma", type=float, default=0.666)
  parser.add_argument('--custom_hparams', type=str)
  return validate_cli


def validate_cli(**args):
  args["custom_hparams"] = split_hparams_string(args["custom_hparams"])
  validate(**args)


def init_inference_parser(parser: ArgumentParser):
  parser.add_argument('--train_name', type=str, required=True)
  parser.add_argument('--wav_path', type=str, required=True)
  parser.add_argument('--custom_checkpoint', type=int)
  parser.add_argument('--sigma', type=float, default=0.666)
  parser.add_argument('--denoiser_strength', type=float, default=0.00)
  parser.add_argument('--custom_hparams', type=str)
  return infer_cli


def infer_cli(**args):
  args["custom_hparams"] = split_hparams_string(args["custom_hparams"])
  infer(**args)


def init_download_parser(parser: ArgumentParser):
  parser.add_argument('--train_name', type=str, default=DEFAULT_WAVEGLOW)
  parser.add_argument('--version', type=int, default=3)
  parser.add_argument('--prep_name', type=str)
  return dl_pretrained
