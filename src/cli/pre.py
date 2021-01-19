from argparse import ArgumentParser

from src.app.pre.ds import (preprocess_arctic, preprocess_custom,
                            preprocess_libritts, preprocess_ljs,
                            preprocess_mailabs, preprocess_thchs,
                            preprocess_thchs_kaldi)
from src.app.pre.inference import (accent_apply, accent_set, add_text,
                                   ipa_convert_text, map_text,
                                   map_to_prep_symbols, normalize_text)
from src.app.pre.mapping import (create_or_update_inference_map_main,
                                 create_or_update_weights_map_main)
from src.app.pre.mel import preprocess_mels
from src.app.pre.plots import plot_mels
from src.app.pre.prepare import prepare_ds
from src.app.pre.text import (preprocess_text, text_convert_to_ipa,
                              text_normalize)
from src.app.pre.tools import remove_silence_plot
from src.app.pre.wav import (preprocess_wavs, wavs_normalize,
                             wavs_remove_silence, wavs_stereo_to_mono,
                             wavs_upsample)
from src.cli.utils import parse_tuple_list, split_hparams_string
from text_utils import EngToIpaMode, Language


def init_preprocess_thchs_parser(parser: ArgumentParser):
  parser.add_argument('--path', type=str, required=True, help='THCHS dataset directory')
  parser.add_argument('--auto_dl', action="store_true")
  parser.add_argument('--ds_name', type=str, required=True, default='thchs')
  return preprocess_thchs


def init_preprocess_ljs_parser(parser: ArgumentParser):
  parser.add_argument('--path', type=str, required=True, help='LJS dataset directory')
  parser.add_argument('--auto_dl', action="store_true")
  parser.add_argument('--ds_name', type=str, required=True, default='ljs')
  return preprocess_ljs

def init_preprocess_mailabs_parser(parser: ArgumentParser):
  parser.add_argument('--path', type=str, required=True, help='M-AILABS dataset directory')
  parser.add_argument('--auto_dl', action="store_true")
  parser.add_argument('--ds_name', type=str, required=True, default='mailabs')
  return preprocess_mailabs


def init_preprocess_arctic_parser(parser: ArgumentParser):
  parser.add_argument('--path', type=str, required=True, help='L2 Arctic dataset directory')
  parser.add_argument('--auto_dl', action="store_true")
  parser.add_argument('--ds_name', type=str, required=True, default='arctic')
  return preprocess_arctic


def init_preprocess_libritts_parser(parser: ArgumentParser):
  parser.add_argument('--path', type=str, required=True, help='LibriTTS dataset directory')
  parser.add_argument('--auto_dl', action="store_true")
  parser.add_argument('--ds_name', type=str, required=True, default='libritts')
  return preprocess_libritts


def init_preprocess_custom_parser(parser: ArgumentParser):
  parser.add_argument('--path', type=str, required=True, help='LibriTTS dataset directory')
  parser.add_argument('--ds_name', type=str, required=True, default='custom')
  parser.set_defaults(auto_dl=False)
  return preprocess_custom


def init_preprocess_thchs_kaldi_parser(parser: ArgumentParser):
  parser.add_argument('--path', type=str, required=True, help='THCHS dataset directory')
  parser.add_argument('--auto_dl', action="store_true")
  parser.add_argument('--ds_name', type=str, required=True, default='thchs_kaldi')
  return preprocess_thchs_kaldi


def init_preprocess_mels_parser(parser: ArgumentParser):
  parser.add_argument('--ds_name', type=str, required=True)
  parser.add_argument('--wav_name', type=str, required=True)
  parser.add_argument('--custom_hparams', type=str)
  return preprocess_mels_cli


def preprocess_mels_cli(**args):
  args["custom_hparams"] = split_hparams_string(args["custom_hparams"])
  preprocess_mels(**args)


def init_mels_plot_parser(parser: ArgumentParser):
  parser.add_argument('--ds_name', type=str, required=True)
  parser.add_argument('--wav_name', type=str, required=True)
  parser.add_argument('--custom_hparams', type=str)
  return plot_mels_cli


def plot_mels_cli(**args):
  args["custom_hparams"] = split_hparams_string(args["custom_hparams"])
  plot_mels(**args)


def init_prepare_ds_parser(parser: ArgumentParser):
  parser.add_argument('--prep_name', type=str, required=True)
  parser.add_argument('--ds_speakers', type=str, required=True)
  parser.add_argument('--ds_text_audio', type=str, required=True)
  return prepare_ds_cli


def prepare_ds_cli(**args):
  args["ds_speakers"] = parse_tuple_list(args["ds_speakers"])
  args["ds_text_audio"] = parse_tuple_list(args["ds_text_audio"])
  prepare_ds(**args)


def init_preprocess_text_parser(parser: ArgumentParser):
  parser.add_argument('--ds_name', type=str, required=True)
  parser.add_argument('--text_name', type=str, required=True)
  return preprocess_text


def init_text_normalize_parser(parser: ArgumentParser):
  parser.add_argument('--ds_name', type=str, required=True)
  parser.add_argument('--orig_text_name', type=str, required=True)
  parser.add_argument('--dest_text_name', type=str, required=True)
  return text_normalize


def init_text_convert_to_ipa_parser(parser: ArgumentParser):
  parser.add_argument('--ds_name', type=str, required=True)
  parser.add_argument('--orig_text_name', type=str, required=True)
  parser.add_argument('--dest_text_name', type=str, required=True)
  parser.add_argument('--ignore_tones', action='store_true')
  parser.add_argument('--ignore_arcs', action='store_true')
  parser.add_argument('--mode', choices=EngToIpaMode,
                      type=EngToIpaMode.__getitem__)
  return text_convert_to_ipa


def init_preprocess_wavs_parser(parser: ArgumentParser):
  parser.add_argument('--ds_name', type=str, required=True)
  parser.add_argument('--wav_name', type=str, required=True)
  return preprocess_wavs


def init_wavs_normalize_parser(parser: ArgumentParser):
  parser.add_argument('--ds_name', type=str, required=True)
  parser.add_argument('--orig_wav_name', type=str, required=True)
  parser.add_argument('--dest_wav_name', type=str, required=True)
  return wavs_normalize


def init_wavs_upsample_parser(parser: ArgumentParser):
  parser.add_argument('--ds_name', type=str, required=True)
  parser.add_argument('--orig_wav_name', type=str, required=True)
  parser.add_argument('--dest_wav_name', type=str, required=True)
  parser.add_argument('--rate', type=int, required=True)
  return wavs_upsample


def init_wavs_stereo_to_mono_parser(parser: ArgumentParser):
  parser.add_argument('--ds_name', type=str, required=True)
  parser.add_argument('--orig_wav_name', type=str, required=True)
  parser.add_argument('--dest_wav_name', type=str, required=True)
  return wavs_stereo_to_mono


def init_wavs_remove_silence_parser(parser: ArgumentParser):
  parser.add_argument('--ds_name', type=str, required=True)
  parser.add_argument('--orig_wav_name', type=str, required=True)
  parser.add_argument('--dest_wav_name', type=str, required=True)
  parser.add_argument('--chunk_size', type=int, required=True)
  parser.add_argument('--threshold_start', type=float, required=True)
  parser.add_argument('--threshold_end', type=float, required=True)
  parser.add_argument('--buffer_start_ms', type=float,
                      help="amount of factors of chunk_size at the beginning and the end should be reserved", required=True)
  parser.add_argument('--buffer_end_ms', type=float,
                      help="amount of factors of chunk_size at the beginning and the end should be reserved", required=True)
  return wavs_remove_silence


def init_wavs_remove_silence_plot_parser(parser: ArgumentParser):
  parser.add_argument('--ds_name', type=str, required=True)
  parser.add_argument('--wav_name', type=str, required=True)
  parser.add_argument('--chunk_size', type=int, required=True)
  parser.add_argument('--entry_id', type=int, help="Keep empty for random entry.")
  parser.add_argument('--threshold_start', type=float, required=True)
  parser.add_argument('--threshold_end', type=float, required=True)
  parser.add_argument('--buffer_start_ms', type=float,
                      help="amount of factors of chunk_size at the beginning and the end should be reserved", required=True)
  parser.add_argument('--buffer_end_ms', type=float,
                      help="amount of factors of chunk_size at the beginning and the end should be reserved", required=True)
  return remove_silence_plot


def init_create_or_update_weights_map_parser(parser: ArgumentParser):
  parser.add_argument('--prep_name', type=str, required=True,
                      help="The prepared name for the model which will be trained.")
  parser.add_argument('--weights_prep_name', type=str, required=True,
                      help="The prepared name of which were used by the pretrained model.")
  parser.add_argument('--template_map', type=str)
  return create_or_update_weights_map_main


def init_create_or_update_inference_map_parser(parser: ArgumentParser):
  parser.add_argument('--prep_name', type=str, required=True)
  parser.add_argument('--template_map', type=str)
  return create_or_update_inference_map_main


def init_add_text_parser(parser: ArgumentParser):
  parser.add_argument('--filepath', type=str, required=False)
  parser.add_argument('--text', type=str, required=False)
  parser.add_argument('--prep_name', type=str, required=True)
  parser.add_argument('--text_name', type=str, required=True)
  parser.add_argument('--lang', choices=Language, type=Language.__getitem__, required=True)
  return add_text


def init_normalize_text_parser(parser: ArgumentParser):
  parser.add_argument('--prep_name', type=str, required=True)
  parser.add_argument('--text_name', type=str, required=True)
  return normalize_text


def init_convert_to_ipa_text_parser(parser: ArgumentParser):
  parser.add_argument('--prep_name', type=str, required=True)
  parser.add_argument('--text_name', type=str, required=True)
  parser.add_argument('--ignore_tones', action='store_true')
  #parser.add_argument('--ignore_arcs', action='store_true')
  parser.add_argument('--mode', choices=EngToIpaMode,
                      type=EngToIpaMode.__getitem__)
  parser.set_defaults(ignore_arcs=True)
  return ipa_convert_text


def init_accent_apply_text_parser(parser: ArgumentParser):
  parser.add_argument('--prep_name', type=str, required=True)
  parser.add_argument('--text_name', type=str, required=True)
  return accent_apply


def init_accent_set_text_parser(parser: ArgumentParser):
  parser.add_argument('--prep_name', type=str, required=True)
  parser.add_argument('--text_name', type=str, required=True)
  parser.add_argument('--accent', type=str, required=True)
  return accent_set


def init_map_text_parser(parser: ArgumentParser):
  parser.add_argument('--prep_name', type=str, required=True)
  parser.add_argument('--text_name', type=str, required=True)
  parser.add_argument('--symbols_map_path', type=str, required=True)
  parser.set_defaults(ignore_arcs=True)
  return map_text


def init_map_to_prep_symbols_parser(parser: ArgumentParser):
  parser.add_argument('--prep_name', type=str, required=True)
  parser.add_argument('--text_name', type=str, required=True)
  parser.set_defaults(ignore_arcs=True)
  return map_to_prep_symbols
