import os
from typing import List, Tuple

from src.cli.pre.io import (get_ds_root_dir, get_filelist_data,
                            get_filelist_dir, get_filelist_speakers_json,
                            get_filelist_symbols_json, get_mel_subdir,
                            get_speakers_json, get_speakers_log_json,
                            get_text_csv, get_text_subdir,
                            get_text_symbol_converter, get_text_symbols_json,
                            get_wav_csv, get_wav_subdir, load_ds_csv,
                            load_mel_csv, load_speaker_json, load_text_csv,
                            load_text_symbol_converter, load_wav_csv,
                            mel_subdir_exists, save_ds_csv, save_filelist,
                            save_filelist_speakers_json,
                            save_filelist_symbol_converter, save_mel_csv,
                            save_speaker_json, save_speaker_log_json,
                            save_text_csv, save_text_symbol_converter,
                            save_text_symbols_json, save_wav_csv,
                            text_subdir_exists, wav_subdir_exists)
from src.core.pre import ljs_preprocess, mels_preprocess
from src.core.pre import merge_ds as merge_ds_core
from src.core.pre import text_convert_to_ipa as text_convert_to_ipa_core
from src.core.pre import text_normalize as text_normalize_core
from src.core.pre import text_preprocess as text_preprocess_core
from src.core.pre import thchs_kaldi_preprocess, thchs_preprocess
from src.core.pre import wavs_normalize as wavs_normalize_core
from src.core.pre import wavs_preprocess as wavs_preprocess_core
from src.core.pre import wavs_remove_silence as wavs_remove_silence_core
from src.core.pre import wavs_upsample as wavs_upsample_core


def preprocess_wavs(base_dir: str, ds_name: str, sub_name: str):
  if wav_subdir_exists(base_dir, ds_name, sub_name):
    print("Already exists.")
  else:
    data = load_ds_csv(base_dir, ds_name)
    #wav_data_dir = get_pre_ds_wav_subname_dir(base_dir, ds_name, sub_name, create=False)
    wav_data = wavs_preprocess_core(data)
    save_wav_csv(base_dir, ds_name, sub_name, wav_data)

def wavs_normalize(base_dir: str, ds_name: str, origin_sub_name: str, destination_sub_name: str):
  if wav_subdir_exists(base_dir, ds_name, destination_sub_name):
    print("Already exists.")
  else:
    data = load_wav_csv(base_dir, ds_name, origin_sub_name)
    wav_data_dir = get_wav_subdir(base_dir, ds_name, destination_sub_name, create=True)
    wav_data = wavs_normalize_core(data, wav_data_dir)
    save_wav_csv(base_dir, ds_name, destination_sub_name, wav_data)

def wavs_upsample(base_dir: str, ds_name: str, origin_sub_name: str, destination_sub_name: str, rate: int):
  if wav_subdir_exists(base_dir, ds_name, destination_sub_name):
    print("Already exists.")
  else:
    data = load_wav_csv(base_dir, ds_name, origin_sub_name)
    wav_data_dir = get_wav_subdir(base_dir, ds_name, destination_sub_name, create=True)
    wav_data = wavs_upsample_core(data, wav_data_dir, rate)
    save_wav_csv(base_dir, ds_name, destination_sub_name, wav_data)

def wavs_remove_silence(base_dir: str, ds_name: str, origin_sub_name: str, destination_sub_name: str, chunk_size: int, threshold_start: float, threshold_end: float, buffer_start_ms: float, buffer_end_ms: float):
  if wav_subdir_exists(base_dir, ds_name, destination_sub_name):
    print("Already exists.")
  else:
    data = load_wav_csv(base_dir, ds_name, origin_sub_name)
    wav_data_dir = get_wav_subdir(base_dir, ds_name, destination_sub_name, create=True)
    wav_data = wavs_remove_silence_core(data, wav_data_dir, chunk_size, threshold_start, threshold_end, buffer_start_ms, buffer_end_ms)
    save_wav_csv(base_dir, ds_name, destination_sub_name, wav_data)

def preprocess_text(base_dir: str, ds_name: str, sub_name: str):
  if text_subdir_exists(base_dir, ds_name, sub_name):
    print("Already exists.")
  else:
    data = load_ds_csv(base_dir, ds_name)
    text_data, conv, all_symbols = text_preprocess_core(data)
    save_text_csv(base_dir, ds_name, sub_name, text_data)
    save_text_symbol_converter(base_dir, ds_name, sub_name, conv)
    save_text_symbols_json(base_dir, ds_name, sub_name, all_symbols)

def text_normalize(base_dir: str, ds_name: str, origin_sub_name: str, destination_sub_name: str):
  if text_subdir_exists(base_dir, ds_name, destination_sub_name):
    print("Already exists.")
  else:
    data = load_text_csv(base_dir, ds_name, origin_sub_name)
    text_data, conv, all_symbols = text_normalize_core(data)
    save_text_csv(base_dir, ds_name, destination_sub_name, text_data)
    save_text_symbol_converter(base_dir, ds_name, destination_sub_name, conv)
    save_text_symbols_json(base_dir, ds_name, destination_sub_name, all_symbols)

def text_convert_to_ipa(base_dir: str, ds_name: str, origin_sub_name: str, destination_sub_name: str, ignore_tones: bool, ignore_arcs: bool):
  if text_subdir_exists(base_dir, ds_name, destination_sub_name):
    print("Already exists.")
  else:
    data = load_text_csv(base_dir, ds_name, origin_sub_name)
    text_data, conv, all_symbols = text_convert_to_ipa_core(data, ignore_tones, ignore_arcs)
    save_text_csv(base_dir, ds_name, destination_sub_name, text_data)
    save_text_symbol_converter(base_dir, ds_name, destination_sub_name, conv)
    save_text_symbols_json(base_dir, ds_name, destination_sub_name, all_symbols)

def _preprocess_ds(base_dir: str, ds_name: str, path: str, auto_dl: bool, preprocess_func):
  ds_path = get_ds_root_dir(base_dir, ds_name, create=False)
  if os.path.isdir(ds_path):
    print("Dataset already processed.")
  else:
    print("Reading data...")
    speakers, speakers_log, ds_data = preprocess_func(path, auto_dl)
    save_speaker_json(base_dir, ds_name, speakers)
    save_speaker_log_json(base_dir, ds_name, speakers_log)
    save_ds_csv(base_dir, ds_name, ds_data)
    print("Dataset processed.")

def preprocess_thchs(base_dir: str, ds_name: str, path: str, auto_dl: bool):
  _preprocess_ds(base_dir, ds_name, path, auto_dl, thchs_preprocess)

def preprocess_thchs_kaldi(base_dir: str, ds_name: str, path: str, auto_dl: bool):
  _preprocess_ds(base_dir, ds_name, path, auto_dl, thchs_kaldi_preprocess)

def preprocess_ljs(base_dir: str, ds_name: str, path: str, auto_dl: bool):
  _preprocess_ds(base_dir, ds_name, path, auto_dl, ljs_preprocess)

def preprocess_mels(base_dir: str, ds_name: str, sub_name: str, custom_hparams: str):
  if mel_subdir_exists(base_dir, ds_name, sub_name):
    print("Already exists.")
  else:
    data = load_wav_csv(base_dir, ds_name, sub_name)
    data_dir = get_mel_subdir(base_dir, ds_name, sub_name, create=True)
    mel_data = mels_preprocess(data, data_dir, custom_hparams)
    save_mel_csv(base_dir, ds_name, sub_name, mel_data)

def prepare_ds(base_dir: str, fl_name: str, ds_speakers: str, ds_text_audio: str):
  if os.path.isdir(get_filelist_data(base_dir, fl_name)):
    print("Already created.")
  else:
    ds_speakers_tuple = _parse_tuple_list(ds_speakers)
    ds_text_audio_tuple = _parse_tuple_list(ds_text_audio)

    datasets = {}
    for ds_name, text_name, audio_name in ds_text_audio_tuple:
      # multiple uses of one ds are not valid
      assert ds_name not in datasets
      
      datasets[ds_name] = (
        load_ds_csv(base_dir, ds_name),
        load_text_csv(base_dir, ds_name, text_name),
        load_wav_csv(base_dir, ds_name, audio_name),
        load_mel_csv(base_dir, ds_name, audio_name),
        load_speaker_json(base_dir, ds_name).get_speakers(),
        load_text_symbol_converter(base_dir, ds_name, text_name)
      )
    
    data, conv, speakers_id_dict = merge_ds_core(datasets, ds_speakers_tuple)

    save_filelist(base_dir, fl_name, data)
    save_filelist_symbol_converter(base_dir, fl_name, conv)
    save_filelist_speakers_json(base_dir, fl_name, speakers_id_dict)

def _parse_tuple_list(tuple_list: str) -> List[Tuple]:
  '''
  Example: [ ['thchs', 'C11', 0], ... ]
  '''
  step1: List[str] = tuple_list.split(';')
  result: List[Tuple] = [tuple(x.split(',')) for x in step1]
  result = list(sorted(set(result)))
  return result

if __name__ == "__main__":
  mode = 5
  
  if mode == 1:
    preprocess_thchs(
      path="/datasets/thchs_wav",
      base_dir="/datasets/models/taco2pt_v2",
      ds_name="thchs",
      auto_dl=True,
    )
  
  elif mode == 2:
    preprocess_ljs(
      base_dir="/datasets/models/taco2pt_v2",
      path="/datasets/LJSpeech-1.1",
      ds_name="ljs",
      auto_dl=True,
    )

  elif mode == 3:
    preprocess_thchs_kaldi(
      base_dir="/datasets/models/taco2pt_v2",
      path="/datasets/THCHS-30",
      ds_name="thchs_kaldi",
      auto_dl=True,
    )

  elif mode == 4:
    preprocess_mels(
      base_dir="/datasets/models/taco2pt_v2",
      ds_name="thchs",
      sub_name="22050kHz_normalized_nosil",
      custom_hparams="",
    )

  elif mode == 5:
    prepare_ds(
      base_dir="/datasets/models/taco2pt_v2",
      fl_name="thchs",
      ds_speakers="thchs,all",
      ds_text_audio="thchs,ipa,22050kHz_normalized_nosil"
    )
    
  elif mode == 6:
    preprocess_text(
      base_dir="/datasets/models/taco2pt_v2",
      ds_name="thchs",
      sub_name="chn",
    )

  elif mode == 7:
    text_convert_to_ipa(
      base_dir="/datasets/models/taco2pt_v2",
      ds_name="thchs",
      origin_sub_name="chn",
      destination_sub_name="ipa",
      ignore_tones=False,
      ignore_arcs=True,
    )

  elif mode == 8:
    preprocess_text(
      base_dir="/datasets/models/taco2pt_v2",
      ds_name="ljs",
      sub_name="en",
    )

  elif mode == 9:
    text_normalize(
      base_dir="/datasets/models/taco2pt_v2",
      ds_name="ljs",
      origin_sub_name="en",
      destination_sub_name="en_norm",
    )

  elif mode == 10:
    text_convert_to_ipa(
      base_dir="/datasets/models/taco2pt_v2",
      ds_name="ljs",
      origin_sub_name="en_norm",
      destination_sub_name="ipa_norm",
      ignore_tones=True,
      ignore_arcs=True,
    )

  elif mode == 11:
    preprocess_wavs(
      base_dir="/datasets/models/taco2pt_v2",
      ds_name="thchs",
      sub_name="16000kHz",
    )

  elif mode == 12:
    wavs_normalize(
      base_dir="/datasets/models/taco2pt_v2",
      ds_name="thchs",
      origin_sub_name="16000kHz",
      destination_sub_name="16000kHz_normalized",
    )

  elif mode == 13:
    wavs_upsample(
      base_dir="/datasets/models/taco2pt_v2",
      ds_name="thchs",
      origin_sub_name="16000kHz_normalized",
      destination_sub_name="22050kHz_normalized",
      rate=22050,
    )

  elif mode == 14:
    wavs_remove_silence(
      base_dir="/datasets/models/taco2pt_v2",
      ds_name="thchs",
      origin_sub_name="22050kHz_normalized",
      destination_sub_name="22050kHz_normalized_nosil",
      threshold_start = -20,
      threshold_end = -30,
      chunk_size = 5,
      buffer_start_ms = 100,
      buffer_end_ms = 150
    )
