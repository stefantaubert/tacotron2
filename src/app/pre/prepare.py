import os
from typing import List, Tuple

from src.app.pre.ds import get_ds_dir, load_ds_csv, load_speaker_json
from src.app.pre.io import get_pre_dir
from src.app.pre.mel import get_mel_dir, load_mel_csv
from src.app.pre.text import (get_text_dir, load_text_csv,
                              load_text_symbol_converter)
from src.app.pre.wav import get_wav_dir, load_wav_csv
from src.core.common import get_subdir
from src.core.pre import (DsData, DsDataList, MelDataList, PreparedData,
                          PreparedDataList, SpeakersDict, SpeakersIdDict,
                          SpeakersLogDict, SymbolConverter, SymbolsDict,
                          TextDataList, WavDataList)
from src.core.pre import merge_ds as merge_ds_core


def _get_prepared_root_dir(base_dir: str, create: bool = False):
  return get_subdir(get_pre_dir(base_dir, create), 'prepared', create)

_prepared_data_csv = "data.csv"
_prepared_speakers_json = "speakers.json"
_prepared_symbols_json = "symbols.json"

def get_prepared_dir(base_dir: str, fl_name: str, create: bool = False):
  return get_subdir(_get_prepared_root_dir(base_dir, create), fl_name, create)

def load_filelist(prep_dir: str) -> PreparedDataList:
  path = os.path.join(prep_dir, _prepared_data_csv)
  return PreparedDataList.load(path)

def save_filelist(prep_dir: str, result: PreparedDataList):
  path = os.path.join(prep_dir, _prepared_data_csv)
  result.save(path)

def load_filelist_speakers_json(prep_dir: str) -> SpeakersIdDict:
  path = os.path.join(prep_dir, _prepared_speakers_json)
  return SpeakersIdDict.load(path)
  
def save_filelist_speakers_json(prep_dir: str, speakers: SpeakersIdDict):
  path = os.path.join(prep_dir, _prepared_speakers_json)
  speakers.save(path)

def load_filelist_symbol_converter(prep_dir: str) -> SymbolConverter:
  path = os.path.join(prep_dir, _prepared_symbols_json)
  return SymbolConverter.load_from_file(path)
  
def save_filelist_symbol_converter(prep_dir: str, data: SymbolConverter):
  path = os.path.join(prep_dir, _prepared_symbols_json)
  data.dump(path)

def _parse_tuple_list(tuple_list: str) -> List[Tuple]:
  """ tuple_list: "a,b;c,d;... """
  step1: List[str] = tuple_list.split(';')
  result: List[Tuple] = [tuple(x.split(',')) for x in step1]
  result = list(sorted(set(result)))
  return result

def prepare_ds(base_dir: str, fl_name: str, ds_speakers: str, ds_text_audio: str):
  prep_dir = get_prepared_dir(base_dir, fl_name)
  if os.path.isdir(prep_dir):
    print("Already created.")
  else:
    os.makedirs(prep_dir)
    ds_speakers_tuple = _parse_tuple_list(ds_speakers)
    ds_text_audio_tuple = _parse_tuple_list(ds_text_audio)

    datasets = {}
    for ds_name, text_name, audio_name in ds_text_audio_tuple:
      # multiple uses of one ds are not valid
      assert ds_name not in datasets
      
      ds_dir = get_ds_dir(base_dir, ds_name)
      text_dir = get_text_dir(ds_dir, text_name)
      wav_dir = get_wav_dir(ds_dir, audio_name)
      mel_dir = get_mel_dir(ds_dir, audio_name)

      datasets[ds_name] = (
        load_ds_csv(ds_dir),
        load_text_csv(text_dir),
        load_wav_csv(wav_dir),
        load_mel_csv(mel_dir),
        load_speaker_json(ds_dir).get_speakers(),
        load_text_symbol_converter(text_dir)
      )
    
    data, conv, speakers_id_dict = merge_ds_core(datasets, ds_speakers_tuple)

    save_filelist(prep_dir, data)
    save_filelist_symbol_converter(prep_dir, conv)
    save_filelist_speakers_json(prep_dir, speakers_id_dict)

if __name__ == "__main__":
  
  prepare_ds(
    base_dir="/datasets/models/taco2pt_v3",
    fl_name="ljs",
    ds_speakers="ljs,all",
    ds_text_audio="ljs,ipa_norm,22050kHz"
  )

  prepare_ds(
    base_dir="/datasets/models/taco2pt_v3",
    fl_name="thchs",
    ds_speakers="thchs,all",
    ds_text_audio="thchs,ipa,22050kHz_normalized_nosil"
  )
