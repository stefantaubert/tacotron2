from argparse import ArgumentParser

from matplotlib import use as use_matplotlib_backend
use_matplotlib_backend("Agg")

from src.pre.silence_removal import init_remove_silence_plot_parser
from src.tacotron.train_io import init_path_parser
from src.tacotron.create_map_template import init_create_map_parser
from src.tacotron.eval_checkpoints import init_eval_checkpoints_parser
from src.tacotron.inference import init_inference_parser as init_taco_inference_parser
from src.tacotron.plot_embeddings import init_plot_emb_parser
from src.tacotron.train import init_train_parser as init_taco_train_parser
from src.tacotron.validate import init_validate_parser as init_taco_validate_parser
from src.waveglow.dl_pretrained import init_download_parser as init_wg_download_parser
from src.waveglow.inference import init_inference_parser as init_wg_inference_parser
from src.waveglow.train import init_train_parser as init_wg_train_parser
from src.waveglow.validate import init_validate_parser as init_wg_validate_parser

from src.pre.wav_pre import init_ljs_parser, init_thchs_kaldi_parser, init_thchs_parser
from src.pre.upsampling import init_upsample_parser
from src.pre.normalize import init_normalize_parser
from src.pre.silence_removal import init_remove_silence_parser
from src.pre.mel_pre import init_calc_mels_parser
from src.pre.text_pre import init_thchs_text_parser, init_ljs_text_pre_parser

def __add_parser_to(subparsers, name: str, init_method):
  parser = subparsers.add_parser(name, help='{} help'.format(name))
  invoke_method = init_method(parser)
  parser.set_defaults(invoke_handler=invoke_method)
  return parser

def __init_parser():
  main_parser = ArgumentParser()
  subparsers = main_parser.add_subparsers(help='sub-command help')

  # only required when automatic name generation
  __add_parser_to(subparsers, "paths", init_path_parser)
  
  __add_parser_to(subparsers, "ljs-wavs", init_ljs_parser)
  __add_parser_to(subparsers, "ljs-text", init_ljs_text_pre_parser)

  __add_parser_to(subparsers, "thchs-wavs", init_thchs_parser)
  __add_parser_to(subparsers, "thchs-text", init_thchs_text_parser)
  __add_parser_to(subparsers, "thchs-kaldi-wavs", init_thchs_kaldi_parser)
  # theoretically not necessary but added to prevent confusion
  __add_parser_to(subparsers, "thchs-kaldi-text", init_thchs_text_parser)

  __add_parser_to(subparsers, "normalize", init_normalize_parser)
  __add_parser_to(subparsers, "upsample", init_upsample_parser)
  __add_parser_to(subparsers, "remove-silence", init_remove_silence_parser)
  __add_parser_to(subparsers, "calc-mels", init_calc_mels_parser)

  # Waveglow
  __add_parser_to(subparsers, "waveglow-dl", init_wg_download_parser)
  __add_parser_to(subparsers, "waveglow-train", init_wg_train_parser)
  __add_parser_to(subparsers, "waveglow-validate", init_wg_validate_parser)
  __add_parser_to(subparsers, "waveglow-infer", init_wg_inference_parser)

  # Tacotron
  __add_parser_to(subparsers, "tacotron-train", init_taco_train_parser)
  __add_parser_to(subparsers, "tacotron-validate", init_taco_validate_parser)
  __add_parser_to(subparsers, "tacotron-infer", init_taco_inference_parser)
  
  # Tools
  __add_parser_to(subparsers, "create-map", init_create_map_parser)
  __add_parser_to(subparsers, "eval-checkpoints", init_eval_checkpoints_parser)
  __add_parser_to(subparsers, "plot-embeddings", init_plot_emb_parser)
  __add_parser_to(subparsers, "remove-silence-plot", init_remove_silence_plot_parser)

  return main_parser

def __process_args(args):
  params = vars(args)
  invoke_handler = params.pop("invoke_handler")
  invoke_handler(**params)

if __name__ == "__main__":
  main_parser = __init_parser()
  
  args = main_parser.parse_args()
  #args = main_parser.parse_args("ljs-text --base_dir=/datasets/models/taco2pt_v2 --mel_name=ljs --ds_name=test_ljs --convert_to_ipa".split())

  __process_args(args)