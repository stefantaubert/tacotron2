import os
from argparse import ArgumentParser
from typing import List, Tuple

from src.cli.pre.ds import load_ds_csv, load_speaker_json
from src.cli.pre.mel import load_mel_csv
from src.cli.pre.paths import (get_filelist_data, get_filelist_dir,
                               get_filelist_speakers_json,
                               get_filelist_symbols_json)
from src.cli.pre.text import load_text_csv, load_text_symbol_converter
from src.cli.pre.wav import load_wav_csv
from src.core.pre import (PreparedData, PreparedDataList, SpeakersIdDict,
                          merge_ds)
from src.text.symbol_converter import SymbolConverter

#region IO

def load_filelist(base_dir: str, fl_name: str) -> PreparedDataList:
  result_path = get_filelist_data(base_dir, fl_name)
  return PreparedDataList.load(result_path)
  
def _save_filelist(base_dir: str, fl_name: str, result: PreparedDataList):
  result_path = get_filelist_data(base_dir, fl_name)
  result.save(result_path)

def load_filelist_speakers_json(base_dir: str, fl_name: str) -> SpeakersIdDict:
  speakers_path = get_filelist_speakers_json(base_dir, fl_name)
  return SpeakersIdDict.load(speakers_path)
  
def _save_filelist_speakers_json(base_dir: str, fl_name: str, speakers: SpeakersIdDict):
  speakers_path = get_filelist_speakers_json(base_dir, fl_name)
  speakers.save(speakers_path)

def load_filelist_symbol_converter(base_dir: str, fl_name: str) -> SymbolConverter:
  data_path = get_filelist_symbols_json(base_dir, fl_name)
  return SymbolConverter.load_from_file(data_path)
  
def _save_filelist_symbol_converter(base_dir: str, fl_name: str, data: SymbolConverter):
  data_path = get_filelist_symbols_json(base_dir, fl_name)
  data.dump(data_path)

#endregion

def parse_tuple_list(tuple_list: str) -> List[Tuple]:
  '''
  Example: [ ['thchs', 'C11', 0], ... ]
  '''
  step1: List[str] = tuple_list.split(';')
  result: List[Tuple] = [tuple(x.split(',')) for x in step1]
  result = list(sorted(set(result)))
  return result

def init_merge_ds_parser(parser: ArgumentParser):
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--fl_name', type=str, required=True)
  parser.add_argument('--ds_speakers', type=str, required=True)
  parser.add_argument('--ds_text_audio', type=str, required=True)
  return _preprocess

def _preprocess(base_dir: str, fl_name: str, ds_speakers: str, ds_text_audio: str):
  if os.path.isdir(get_filelist_data(base_dir, fl_name)):
    print("Already created.")
  else:
    ds_speakers_tuple = parse_tuple_list(ds_speakers)
    ds_text_audio_tuple = parse_tuple_list(ds_text_audio)

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
    
    data, conv, speakers_id_dict = merge_ds(datasets, ds_speakers_tuple)

    _save_filelist(base_dir, fl_name, data)
    _save_filelist_symbol_converter(base_dir, fl_name, conv)
    _save_filelist_speakers_json(base_dir, fl_name, speakers_id_dict)

if __name__ == "__main__":
  _preprocess(
    base_dir="/datasets/models/taco2pt_v2",
    fl_name="thchs",
    ds_speakers="thchs,all",
    ds_text_audio="thchs,ipa,22050kHz_normalized_nosil"
  )
