from argparse import ArgumentParser

from matplotlib import use as use_matplotlib_backend
use_matplotlib_backend("Agg")

from src.cli.pre import (init_prepare_ds_parser, init_preprocess_ljs_parser,
                         init_preprocess_mels_parser,
                         init_preprocess_text_parser,
                         init_preprocess_thchs_kaldi_parser,
                         init_preprocess_thchs_parser,
                         init_preprocess_wavs_parser,
                         init_text_convert_to_ipa_parser,
                         init_text_normalize_parser,
                         init_wavs_normalize_parser,
                         init_wavs_remove_silence_parser,
                         init_wavs_upsample_parser)
from src.cli.tacotron import \
    init_continue_train_parser as init_taco_continue_train_parser
from src.cli.tacotron import init_inference_parser as init_taco_infer_parser
from src.cli.tacotron import init_train_parser as init_taco_train_parser
from src.cli.tacotron import init_validate_parser as init_taco_val_parser
from src.cli.waveglow import \
    init_continue_train_parser as init_wg_continue_train_parser
from src.cli.waveglow import init_inference_parser as init_wg_infer_parser
from src.cli.waveglow import init_train_parser as init_wg_train_parser
from src.cli.waveglow import init_validate_parser as init_wg_val_parser, init_download_parser as init_wg_dl_parser


def _add_parser_to(subparsers, name: str, init_method):
  parser = subparsers.add_parser(name, help='{} help'.format(name))
  invoke_method = init_method(parser)
  parser.set_defaults(invoke_handler=invoke_method)
  return parser

def _init_parser():
  main_parser = ArgumentParser()
  subparsers = main_parser.add_subparsers(help='sub-command help')

  # only required when automatic name generation
  #__add_parser_to(subparsers, "paths", init_path_parser)
  _add_parser_to(subparsers, "preprocess-ljs", init_preprocess_ljs_parser)
  _add_parser_to(subparsers, "preprocess-thchs", init_preprocess_thchs_parser)
  _add_parser_to(subparsers, "preprocess-thchs-kaldi", init_preprocess_thchs_kaldi_parser)
  
  _add_parser_to(subparsers, "preprocess-wavs", init_preprocess_wavs_parser)
  _add_parser_to(subparsers, "wavs-normalize", init_wavs_normalize_parser)
  _add_parser_to(subparsers, "wavs-upsample", init_wavs_upsample_parser)
  _add_parser_to(subparsers, "wavs-remove-silence", init_wavs_remove_silence_parser)

  _add_parser_to(subparsers, "preprocess-text", init_preprocess_text_parser)
  _add_parser_to(subparsers, "text-normalize", init_text_normalize_parser)
  _add_parser_to(subparsers, "text-ipa", init_text_convert_to_ipa_parser)

  _add_parser_to(subparsers, "preprocess-mels", init_preprocess_mels_parser)

  _add_parser_to(subparsers, "prepare-ds", init_prepare_ds_parser)

  # Waveglow
  _add_parser_to(subparsers, "waveglow-dl", init_wg_dl_parser)
  _add_parser_to(subparsers, "waveglow-train", init_wg_train_parser)
  _add_parser_to(subparsers, "waveglow-continue-train", init_wg_continue_train_parser)
  _add_parser_to(subparsers, "waveglow-validate", init_wg_val_parser)
  _add_parser_to(subparsers, "waveglow-infer", init_wg_infer_parser)

  # Tacotron
  _add_parser_to(subparsers, "tacotron-train", init_taco_train_parser)
  _add_parser_to(subparsers, "tacotron-continue-train", init_taco_continue_train_parser)
  _add_parser_to(subparsers, "tacotron-validate", init_taco_val_parser)
  _add_parser_to(subparsers, "tacotron-infer", init_taco_infer_parser)
  
  # # Tools
  # _add_parser_to(subparsers, "create-map", init_create_map_parser)
  # _add_parser_to(subparsers, "eval-checkpoints", init_eval_checkpoints_parser)
  # _add_parser_to(subparsers, "plot-embeddings", init_plot_emb_parser)
  # _add_parser_to(subparsers, "remove-silence-plot", init_remove_silence_plot_parser)

  return main_parser

def _process_args(args):
  params = vars(args)
  invoke_handler = params.pop("invoke_handler")
  invoke_handler(**params)

if __name__ == "__main__":
  main_parser = _init_parser()
  
  args = main_parser.parse_args()
  #args = main_parser.parse_args("ljs-text --base_dir=/datasets/models/taco2pt_v2 --mel_name=ljs --ds_name=test_ljs --convert_to_ipa".split())

  _process_args(args)
