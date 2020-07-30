from argparse import ArgumentParser

from matplotlib import use as use_matplotlib_backend
use_matplotlib_backend("Agg")

from src.common.audio.remove_silence import init_remove_silence_plot_parser
from src.paths import init_path_parser
from src.pre.ljs import init_calc_mels_parser as init_ljs_calc_mels_parser
from src.pre.ljs import init_download_parser as init_ljs_download_parser
from src.pre.ljs import init_text_pre_parser as init_ljs_text_pre_parser
from src.pre.thchs import init_calc_mels_parser as init_thchs_calc_mels_parser
from src.pre.thchs import init_download_parser as init_thchs_download_parser
from src.pre.thchs import init_remove_silence_parser as init_thchs_remove_silence_parser
from src.pre.thchs import init_upsample_parser as init_thchs_upsample_parser
from src.pre.thchs import init_text_pre_parser as init_thchs_text_pre_parser
from src.pre.thchs_kaldi import init_calc_mels_parser as init_thchs_kaldi_calc_mels_parser
from src.pre.thchs_kaldi import init_download_parser as init_thchs_kaldi_download_parser
from src.pre.thchs_kaldi import init_remove_silence_parser as init_thchs_kaldi_remove_silence_parser
from src.pre.thchs_kaldi import init_upsample_parser as init_thchs_kaldi_upsample_parser
from src.pre.thchs_kaldi import init_text_pre_parser as init_thchs_kaldi_text_pre_parser
from src.tacotron.create_map_template import init_create_map_parser
from src.tacotron.eval_checkpoints import init_eval_checkpoints_parser
from src.tacotron.inference import init_inference_parser as init_taco_inference_parser
from src.tacotron.plot_embeddings import init_plot_emb_parser
from src.tacotron.train import init_train_parser as init_taco_train_parser
from src.tacotron.validate import init_validate_parser as init_taco_validate_parser
from src.waveglow.converter.convert import init_converter_parser as init_wg_converter_parser
from src.waveglow.dl_pretrained import init_download_parser as init_wg_download_parser
from src.waveglow.inference import init_inference_parser as init_wg_inference_parser
from src.waveglow.train import init_train_parser as init_wg_train_parser
from src.waveglow.validate import init_validate_parser as init_wg_validate_parser

def __add_parser_to(subparsers, name: str, init_method):
  parser = subparsers.add_parser(name, help='{} help'.format(name))
  invoke_method = init_method(parser)
  parser.set_defaults(invoke_handler=invoke_method)
  return parser

def __init_parser():
  main_parser = ArgumentParser()
  subparsers = main_parser.add_subparsers(help='sub-command help')

  __add_parser_to(subparsers, "paths", init_path_parser)

  __add_parser_to(subparsers, "ljs-dl", init_ljs_download_parser)
  __add_parser_to(subparsers, "ljs-mels", init_ljs_calc_mels_parser)
  __add_parser_to(subparsers, "ljs-text", init_ljs_text_pre_parser)

  __add_parser_to(subparsers, "thchs-dl", init_thchs_download_parser)
  __add_parser_to(subparsers, "thchs-upsample", init_thchs_upsample_parser)
  __add_parser_to(subparsers, "thchs-remove-silence", init_thchs_remove_silence_parser)
  __add_parser_to(subparsers, "thchs-mels", init_thchs_calc_mels_parser)
  __add_parser_to(subparsers, "thchs-text", init_thchs_text_pre_parser)

  __add_parser_to(subparsers, "thchs-kaldi-dl", init_thchs_kaldi_download_parser)
  __add_parser_to(subparsers, "thchs-kaldi-upsample", init_thchs_kaldi_upsample_parser)
  __add_parser_to(subparsers, "thchs-kaldi-remove-silence", init_thchs_kaldi_remove_silence_parser)
  __add_parser_to(subparsers, "thchs-kaldi-mels", init_thchs_kaldi_calc_mels_parser)
  __add_parser_to(subparsers, "thchs-kaldi-text", init_thchs_kaldi_text_pre_parser)

  __add_parser_to(subparsers, "create-map", init_create_map_parser)
  __add_parser_to(subparsers, "eval-checkpoints", init_eval_checkpoints_parser)
  __add_parser_to(subparsers, "plot-embeddings", init_plot_emb_parser)
  __add_parser_to(subparsers, "remove-silence", init_remove_silence_plot_parser)

  __add_parser_to(subparsers, "tacotron-train", init_taco_train_parser)
  __add_parser_to(subparsers, "tacotron-validate", init_taco_validate_parser)
  __add_parser_to(subparsers, "tacotron-infer", init_taco_inference_parser)

  __add_parser_to(subparsers, "waveglow-convert", init_wg_converter_parser)
  __add_parser_to(subparsers, "waveglow-dl", init_wg_download_parser)
  __add_parser_to(subparsers, "waveglow-train", init_wg_train_parser)
  __add_parser_to(subparsers, "waveglow-validate", init_wg_validate_parser)
  __add_parser_to(subparsers, "waveglow-infer", init_wg_inference_parser)
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