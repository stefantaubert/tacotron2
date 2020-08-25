from argparse import ArgumentParser
from src.app import wg_train, wg_continue_train, wg_infer, wg_validate, wg_dl_pretrained
from src.cli.utils import parse_tuple_list
from typing import Optional

def init_train_parser(parser: ArgumentParser):
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--train_name', type=str, required=True)
  parser.add_argument('--prep_name', type=str, required=True)
  parser.add_argument('--test_size', type=float, default=0.001)
  parser.add_argument('--validation_size', type=float, default=0.1)
  parser.add_argument('--split_seed', type=int, default=1234)
  parser.add_argument('--hparams', type=str)
  return wg_train

def init_continue_train_parser(parser: ArgumentParser):
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--train_name', type=str, required=True)
  parser.add_argument('--hparams', type=str)
  return wg_continue_train

def init_validate_parser(parser: ArgumentParser):
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--train_name', type=str, required=True)
  parser.add_argument('--entry_id', type=int, help="Utterance id or nothing if random")
  parser.add_argument('--ds_speaker', type=str, help="ds_name,speaker_name")
  parser.add_argument('--ds', type=str, help="Choose if validation- or testset should be taken.", choices=["val", "test"], default="val")
  parser.add_argument('--custom_checkpoint', type=int, default=0)
  parser.add_argument("--denoiser_strength", default=0.00, type=float, help='Removes model bias.')
  parser.add_argument("--sigma", type=float, default=0.666)
  parser.add_argument('--sampling_rate', type=float, default=22050)
  return wg_validate

def _wg_validate(base_dir: str, train_name: str, entry_id: Optional[int], ds_speaker: Optional[str], ds: str, custom_checkpoint: Optional[int], sigma: float, denoiser_strength: float, sampling_rate: float):
  wg_validate(base_dir, train_name, entry_id, parse_tuple_list(ds_speaker), ds, custom_checkpoint, sigma, denoiser_strength, sampling_rate)

def init_inference_parser(parser):
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--train_name', type=str, required=True)
  parser.add_argument('--wav_path', type=str, required=True)
  parser.add_argument('--custom_checkpoint', type=int)
  parser.add_argument('--sigma', type=float, default=0.666)
  parser.add_argument('--denoiser_strength', type=float, default=0.00)
  parser.add_argument('--sampling_rate', type=float, default=22050)
  return wg_infer

def init_download_parser(parser: ArgumentParser):
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--train_name', type=str, default="pretrained")
  return wg_dl_pretrained
