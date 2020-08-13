from argparse import ArgumentParser

from matplotlib import use as use_matplotlib_backend
use_matplotlib_backend("Agg")

from src.cli.pre import (init_ljs_parser, init_mel_parser,
                         init_text_ipa_parser, init_text_normalize_parser,
                         init_text_parser, init_thchs_kaldi_parser,
                         init_thchs_parser, init_wav_normalize_parser, init_merge_ds_parser,
                         init_wav_parser, init_wav_remove_silence_parser,
                         init_wav_upsample_parser)
from src.cli.tacotron import init_taco_continue_train_parser, init_taco_train_parser

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
  _add_parser_to(subparsers, "parse-ljs", init_ljs_parser)
  _add_parser_to(subparsers, "parse-thchs", init_thchs_parser)
  _add_parser_to(subparsers, "parse-thchs-kaldi", init_thchs_kaldi_parser)
  
  _add_parser_to(subparsers, "parse-wavs", init_wav_parser)
  _add_parser_to(subparsers, "wavs-normalize", init_wav_normalize_parser)
  _add_parser_to(subparsers, "wavs-upsample", init_wav_upsample_parser)
  _add_parser_to(subparsers, "wavs-remove-silence", init_wav_remove_silence_parser)

  _add_parser_to(subparsers, "parse-text", init_text_parser)
  _add_parser_to(subparsers, "text-normalize", init_text_normalize_parser)
  _add_parser_to(subparsers, "text-ipa", init_text_ipa_parser)

  _add_parser_to(subparsers, "parse-mels", init_mel_parser)

  _add_parser_to(subparsers, "prepare-ds", init_merge_ds_parser)


  # # Waveglow
  # _add_parser_to(subparsers, "waveglow-dl", init_wg_download_parser)
  # _add_parser_to(subparsers, "waveglow-train", init_wg_train_parser)
  # _add_parser_to(subparsers, "waveglow-validate", init_wg_validate_parser)
  # _add_parser_to(subparsers, "waveglow-infer", init_wg_inference_parser)

  # Tacotron
  _add_parser_to(subparsers, "tacotron-train", init_taco_train_parser)
  _add_parser_to(subparsers, "tacotron-continue-train", init_taco_continue_train_parser)
  # _add_parser_to(subparsers, "tacotron-validate", init_taco_validate_parser)
  # _add_parser_to(subparsers, "tacotron-infer", init_taco_inference_parser)
  
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
