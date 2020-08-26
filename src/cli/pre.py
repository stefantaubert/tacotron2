from argparse import ArgumentParser

from src.app import (preprocess_mels, prepare_ds, preprocess_ljs,
                              preprocess_thchs, preprocess_thchs_kaldi,
                              text_convert_to_ipa, text_normalize,
                              preprocess_text, wavs_normalize, preprocess_wavs, wavs_remove_silence_plot,
                              wavs_remove_silence, wavs_upsample, create_weights_map, create_inference_map)
from src.cli.utils import parse_tuple_list

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
  parser.add_argument('--wav_name', type=str, required=True)
  parser.add_argument('--custom_hparams', type=str)
  return preprocess_mels

def init_prepare_ds_parser(parser: ArgumentParser):
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--prep_name', type=str, required=True)
  parser.add_argument('--ds_speakers', type=str, required=True)
  parser.add_argument('--ds_text_audio', type=str, required=True)
  return _prepare_ds_cli

def _prepare_ds_cli(base_dir: str, prep_name: str, ds_speakers: str, ds_text_audio: str):
  prepare_ds(base_dir=base_dir, prep_name=prep_name, ds_speakers=parse_tuple_list(ds_speakers), ds_text_audio=parse_tuple_list(ds_text_audio))

def init_preprocess_text_parser(parser: ArgumentParser):
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--ds_name', type=str, required=True)
  parser.add_argument('--text_name', type=str, required=True)
  return preprocess_text

def init_text_normalize_parser(parser: ArgumentParser):
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--ds_name', type=str, required=True)
  parser.add_argument('--orig_text_name', type=str, required=True)
  parser.add_argument('--dest_text_name', type=str, required=True)
  return text_normalize

def init_text_convert_to_ipa_parser(parser: ArgumentParser):
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--ds_name', type=str, required=True)
  parser.add_argument('--orig_text_name', type=str, required=True)
  parser.add_argument('--dest_text_name', type=str, required=True)
  parser.add_argument('--ignore_tones', action='store_true')
  parser.add_argument('--ignore_arcs', action='store_true')
  return text_convert_to_ipa

def init_preprocess_wavs_parser(parser: ArgumentParser):
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--ds_name', type=str, required=True)
  parser.add_argument('--wav_name', type=str, required=True)
  return preprocess_wavs

def init_wavs_normalize_parser(parser: ArgumentParser):
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--ds_name', type=str, required=True)
  parser.add_argument('--orig_wav_name', type=str, required=True)
  parser.add_argument('--dest_wav_name', type=str, required=True)
  return wavs_normalize

def init_wavs_upsample_parser(parser: ArgumentParser):
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--ds_name', type=str, required=True)
  parser.add_argument('--orig_wav_name', type=str, required=True)
  parser.add_argument('--dest_wav_name', type=str, required=True)
  parser.add_argument('--rate', type=int, required=True)
  return wavs_upsample

def init_wavs_remove_silence_parser(parser: ArgumentParser):
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--ds_name', type=str, required=True)
  parser.add_argument('--orig_wav_name', type=str, required=True)
  parser.add_argument('--dest_wav_name', type=str, required=True)
  parser.add_argument('--chunk_size', type=int, required=True)
  parser.add_argument('--threshold_start', type=float, required=True)
  parser.add_argument('--threshold_end', type=float, required=True)
  parser.add_argument('--buffer_start_ms', type=float, help="amount of factors of chunk_size at the beginning and the end should be reserved", required=True)
  parser.add_argument('--buffer_end_ms', type=float, help="amount of factors of chunk_size at the beginning and the end should be reserved", required=True)
  return wavs_remove_silence

def init_wavs_remove_silence_plot_parser(parser: ArgumentParser):
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--ds_name', type=str, required=True)
  parser.add_argument('--wav_name', type=str, required=True)
  parser.add_argument('--chunk_size', type=int, required=True)
  parser.add_argument('--entry_id', type=int, help="Keep empty for random entry.")
  parser.add_argument('--threshold_start', type=float, required=True)
  parser.add_argument('--threshold_end', type=float, required=True)
  parser.add_argument('--buffer_start_ms', type=float, help="amount of factors of chunk_size at the beginning and the end should be reserved", required=True)
  parser.add_argument('--buffer_end_ms', type=float, help="amount of factors of chunk_size at the beginning and the end should be reserved", required=True)
  return wavs_remove_silence_plot

def init_create_weights_map_parser(parser: ArgumentParser):
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--orig_prep_name', type=str, required=True)
  parser.add_argument('--dest_prep_name', type=str, required=True)
  parser.add_argument('--existing_map', type=str)
  parser.add_argument('--dest_dir', type=str, default="maps/weights")
  return create_weights_map

def init_create_inference_map_parser(parser: ArgumentParser):
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--prep_name', type=str, required=True)
  parser.add_argument('--corpora', type=str, required=True)
  parser.add_argument('--is_ipa', action='store_true')
  parser.add_argument('--ignore_tones', action='store_true')
  parser.add_argument('--ignore_arcs', action='store_true')
  parser.add_argument('--existing_map', type=str)
  parser.add_argument('--dest_dir', type=str, default="maps/inference")
  return create_inference_map
