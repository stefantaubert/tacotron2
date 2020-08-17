from argparse import ArgumentParser

from src.cli.pre.main import (preprocess_mels, prepare_ds, preprocess_ljs,
                              preprocess_thchs, preprocess_thchs_kaldi,
                              text_convert_to_ipa, text_normalize,
                              preprocess_text, wavs_normalize, preprocess_wavs,
                              wavs_remove_silence, wavs_upsample)


def init_preprocess_thchs_parser(parser: ArgumentParser):
  parser.add_argument('--path', type=str, required=True, help='THCHS dataset directory')
  parser.add_argument('--auto_dl', action="store_true")
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--ds_name', type=str, required=True, default='thchs')
  return preprocess_thchs

def init_preprocess_ljs_parser(parser: ArgumentParser):
  parser.add_argument('--path', type=str, required=True, help='LJS dataset directory')
  parser.add_argument('--auto_dl', action="store_true")
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--ds_name', type=str, required=True, default='ljs')
  return preprocess_ljs

def init_preprocess_thchs_kaldi_parser(parser: ArgumentParser):
  parser.add_argument('--path', type=str, required=True, help='THCHS dataset directory')
  parser.add_argument('--auto_dl', action="store_true")
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--ds_name', type=str, required=True, default='thchs_kaldi')
  return preprocess_thchs_kaldi

def init_preprocess_mels_parser(parser: ArgumentParser):
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--ds_name', type=str, required=True)
  parser.add_argument('--sub_name', type=str, required=True)
  parser.add_argument('--custom_hparams', type=str)
  return preprocess_mels

def init_prepare_ds_parser(parser: ArgumentParser):
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--fl_name', type=str, required=True)
  parser.add_argument('--ds_speakers', type=str, required=True)
  parser.add_argument('--ds_text_audio', type=str, required=True)
  return prepare_ds

def init_preprocess_text_parser(parser: ArgumentParser):
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--ds_name', type=str, required=True)
  parser.add_argument('--sub_name', type=str, required=True)
  return preprocess_text

def init_text_normalize_parser(parser: ArgumentParser):
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--ds_name', type=str, required=True)
  parser.add_argument('--origin_sub_name', type=str, required=True)
  parser.add_argument('--destination_sub_name', type=str, required=True)
  return text_normalize

def init_text_convert_to_ipa_parser(parser: ArgumentParser):
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--ds_name', type=str, required=True)
  parser.add_argument('--origin_sub_name', type=str, required=True)
  parser.add_argument('--destination_sub_name', type=str, required=True)
  parser.add_argument('--ignore_tones', action='store_true')
  parser.add_argument('--ignore_arcs', action='store_true')
  return text_convert_to_ipa

def init_preprocess_wavs_parser(parser: ArgumentParser):
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--ds_name', type=str, required=True)
  parser.add_argument('--sub_name', type=str, required=True)
  return preprocess_wavs

def init_wavs_normalize_parser(parser: ArgumentParser):
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--ds_name', type=str, required=True)
  parser.add_argument('--origin_sub_name', type=str, required=True)
  parser.add_argument('--destination_sub_name', type=str, required=True)
  return wavs_normalize

def init_wavs_upsample_parser(parser: ArgumentParser):
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--ds_name', type=str, required=True)
  parser.add_argument('--origin_sub_name', type=str, required=True)
  parser.add_argument('--destination_sub_name', type=str, required=True)
  parser.add_argument('--rate', type=int, required=True)
  return wavs_upsample

def init_wavs_remove_silence_parser(parser: ArgumentParser):
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--ds_name', type=str, required=True)
  parser.add_argument('--origin_sub_name', type=str, required=True)
  parser.add_argument('--destination_sub_name', type=str, required=True)
  parser.add_argument('--chunk_size', type=int, required=True)
  parser.add_argument('--threshold_start', type=float, required=True)
  parser.add_argument('--threshold_end', type=float, required=True)
  parser.add_argument('--buffer_start_ms', type=float, help="amount of factors of chunk_size at the beginning and the end should be reserved", required=True)
  parser.add_argument('--buffer_end_ms', type=float, help="amount of factors of chunk_size at the beginning and the end should be reserved", required=True)
  return wavs_remove_silence
