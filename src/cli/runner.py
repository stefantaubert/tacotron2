from argparse import ArgumentParser

from src.cli.pre import (init_accent_apply_text_parser,
                         init_accent_set_text_parser, init_add_text_parser,
                         init_convert_to_ipa_text_parser,
                         init_create_or_update_inference_map_parser,
                         init_create_or_update_weights_map_parser,
                         init_map_text_parser, init_map_to_prep_symbols_parser,
                         init_mels_plot_parser, init_normalize_text_parser,
                         init_prepare_ds_parser, init_preprocess_arctic_parser,
                         init_preprocess_custom_parser,
                         init_preprocess_libritts_parser,
                         init_preprocess_ljs_parser,
                         init_preprocess_mailabs_parser,
                         init_preprocess_mels_parser,
                         init_preprocess_text_parser,
                         init_preprocess_thchs_kaldi_parser,
                         init_preprocess_thchs_parser,
                         init_preprocess_wavs_parser,
                         init_text_convert_to_ipa_parser,
                         init_text_normalize_parser,
                         init_wavs_normalize_parser,
                         init_wavs_remove_silence_parser,
                         init_wavs_remove_silence_plot_parser,
                         init_wavs_stats_parser,
                         init_wavs_stereo_to_mono_parser,
                         init_wavs_upsample_parser)
from src.cli.tacotron import \
    init_continue_train_parser as init_taco_continue_train_parser
from src.cli.tacotron import \
    init_eval_checkpoints_parser as init_taco_eval_checkpoints_parser
from src.cli.tacotron import init_inference_parser as init_taco_infer_parser
from src.cli.tacotron import init_plot_emb_parser as init_taco_plot_emb_parser
from src.cli.tacotron import init_restore_parser as init_taco_restore_parser
from src.cli.tacotron import init_train_parser as init_taco_train_parser
from src.cli.tacotron import init_validate_parser as init_taco_val_parser
from src.cli.utils import add_base_dir
from src.cli.waveglow import \
    init_continue_train_parser as init_wg_continue_train_parser
from src.cli.waveglow import init_download_parser as init_wg_dl_parser
from src.cli.waveglow import init_inference_parser as init_wg_infer_parser
from src.cli.waveglow import init_train_parser as init_wg_train_parser
from src.cli.waveglow import init_validate_parser as init_wg_val_parser


def _add_parser_to(subparsers, name: str, init_method):
  parser = subparsers.add_parser(name, help=f"{name} help")
  invoke_method = init_method(parser)
  parser.set_defaults(invoke_handler=invoke_method)
  add_base_dir(parser)
  return parser


def _init_parser():
  result = ArgumentParser()
  subparsers = result.add_subparsers(help='sub-command help')

  _add_parser_to(subparsers, "preprocess-custom", init_preprocess_custom_parser)
  _add_parser_to(subparsers, "preprocess-ljs", init_preprocess_ljs_parser)
  _add_parser_to(subparsers, "preprocess-mailabs", init_preprocess_mailabs_parser)
  _add_parser_to(subparsers, "preprocess-arctic", init_preprocess_arctic_parser)
  _add_parser_to(subparsers, "preprocess-libritts", init_preprocess_libritts_parser)
  _add_parser_to(subparsers, "preprocess-thchs", init_preprocess_thchs_parser)
  _add_parser_to(subparsers, "preprocess-thchs-kaldi", init_preprocess_thchs_kaldi_parser)

  _add_parser_to(subparsers, "preprocess-wavs", init_preprocess_wavs_parser)
  _add_parser_to(subparsers, "wavs-stats", init_wavs_stats_parser)
  _add_parser_to(subparsers, "wavs-normalize", init_wavs_normalize_parser)
  _add_parser_to(subparsers, "wavs-resample", init_wavs_upsample_parser)
  _add_parser_to(subparsers, "wavs-stereo-to-mono", init_wavs_stereo_to_mono_parser)
  _add_parser_to(subparsers, "wavs-remove-silence", init_wavs_remove_silence_parser)
  _add_parser_to(subparsers, "wavs-remove-silence-plot", init_wavs_remove_silence_plot_parser)

  _add_parser_to(subparsers, "preprocess-text", init_preprocess_text_parser)
  _add_parser_to(subparsers, "text-normalize", init_text_normalize_parser)
  _add_parser_to(subparsers, "text-ipa", init_text_convert_to_ipa_parser)

  _add_parser_to(subparsers, "preprocess-mels", init_preprocess_mels_parser)
  # is also possible without preprocess mels first
  _add_parser_to(subparsers, "mels-plot", init_mels_plot_parser)

  _add_parser_to(subparsers, "prepare-ds", init_prepare_ds_parser)
  _add_parser_to(subparsers, "prepare-text-add", init_add_text_parser)
  _add_parser_to(subparsers, "prepare-text-normalize", init_normalize_text_parser)
  _add_parser_to(subparsers, "prepare-text-to-ipa", init_convert_to_ipa_text_parser)
  _add_parser_to(subparsers, "prepare-text-set-accent", init_accent_set_text_parser)
  _add_parser_to(subparsers, "prepare-text-apply-accents", init_accent_apply_text_parser)
  _add_parser_to(subparsers, "prepare-text-map", init_map_text_parser)
  _add_parser_to(subparsers, "prepare-text-automap", init_map_to_prep_symbols_parser)
  _add_parser_to(subparsers, "prepare-weights-map", init_create_or_update_weights_map_parser)
  _add_parser_to(subparsers, "prepare-inference-map", init_create_or_update_inference_map_parser)

  _add_parser_to(subparsers, "waveglow-dl", init_wg_dl_parser)
  _add_parser_to(subparsers, "waveglow-train", init_wg_train_parser)
  _add_parser_to(subparsers, "waveglow-continue-train", init_wg_continue_train_parser)
  _add_parser_to(subparsers, "waveglow-validate", init_wg_val_parser)
  _add_parser_to(subparsers, "waveglow-infer", init_wg_infer_parser)

  _add_parser_to(subparsers, "tacotron-restore", init_taco_restore_parser)
  _add_parser_to(subparsers, "tacotron-train", init_taco_train_parser)
  _add_parser_to(subparsers, "tacotron-continue-train", init_taco_continue_train_parser)
  _add_parser_to(subparsers, "tacotron-validate", init_taco_val_parser)
  _add_parser_to(subparsers, "tacotron-infer", init_taco_infer_parser)
  _add_parser_to(subparsers, "tacotron-eval-checkpoints", init_taco_eval_checkpoints_parser)
  _add_parser_to(subparsers, "tacotron-plot-embeddings", init_taco_plot_emb_parser)

  return result


def _process_args(args):
  params = vars(args)
  invoke_handler = params.pop("invoke_handler")
  invoke_handler(**params)


if __name__ == "__main__":
  main_parser = _init_parser()

  received_args = main_parser.parse_args()
  #args = main_parser.parse_args("ljs-text --base_dir=/datasets/models/taco2pt_v2 --mel_name=ljs --ds_name=test_ljs --convert_to_ipa".split())

  _process_args(received_args)
