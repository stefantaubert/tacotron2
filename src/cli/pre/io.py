import os

from src.core.common import get_subdir
from src.core.pre import (DsData, DsDataList, MelDataList, PreparedData,
                          PreparedDataList, SpeakersDict, SpeakersIdDict,
                          SpeakersLogDict, SymbolConverter, SymbolsDict,
                          TextDataList, WavDataList)

def _get_pre_dir(base_dir: str, create: bool = False):
  return get_subdir(base_dir, 'pre', create)

#region Filelist

def _get_filelist_root_dir(base_dir: str, create: bool = False):
  return get_subdir(_get_pre_dir(base_dir), 'prepared', create)

def load_filelist(base_dir: str, fl_name: str) -> PreparedDataList:
  result_path = get_filelist_data(base_dir, fl_name)
  return PreparedDataList.load(result_path)

def save_filelist(base_dir: str, fl_name: str, result: PreparedDataList):
  result_path = get_filelist_data(base_dir, fl_name)
  result.save(result_path)

def load_filelist_speakers_json(base_dir: str, fl_name: str) -> SpeakersIdDict:
  speakers_path = get_filelist_speakers_json(base_dir, fl_name)
  return SpeakersIdDict.load(speakers_path)
  
def save_filelist_speakers_json(base_dir: str, fl_name: str, speakers: SpeakersIdDict):
  speakers_path = get_filelist_speakers_json(base_dir, fl_name)
  speakers.save(speakers_path)

def load_filelist_symbol_converter(base_dir: str, fl_name: str) -> SymbolConverter:
  data_path = get_filelist_symbols_json(base_dir, fl_name)
  return SymbolConverter.load_from_file(data_path)
  
def save_filelist_symbol_converter(base_dir: str, fl_name: str, data: SymbolConverter):
  data_path = get_filelist_symbols_json(base_dir, fl_name)
  data.dump(data_path)

def get_filelist_dir(base_dir: str, fl_name: str, create: bool = False):
  return get_subdir(_get_filelist_root_dir(base_dir), fl_name, create)

def get_filelist_speakers_json(base_dir: str, fl_name: str):
  return os.path.join(get_filelist_dir(base_dir, fl_name, True), "speakers.json")

def get_filelist_symbols_json(base_dir: str, fl_name: str):
  return os.path.join(get_filelist_dir(base_dir, fl_name, True), "symbols.json")

def get_filelist_data(base_dir: str, fl_name: str):
  return os.path.join(get_filelist_dir(base_dir, fl_name, True), "data.csv")

#endregion

#region Ds

def get_ds_root_dir(base_dir: str, ds_name: str, create: bool = False):
  return get_subdir(_get_pre_dir(base_dir), ds_name, create)

def load_ds_csv(base_dir: str, ds_name: str) -> DsDataList:
  result_path = get_ds_csv(base_dir, ds_name)
  return DsDataList.load(result_path)
  
def save_ds_csv(base_dir: str, ds_name: str, result: DsDataList):
  result_path = get_ds_csv(base_dir, ds_name)
  result.save(result_path)

def load_speaker_json(base_dir: str, ds_name: str) -> SpeakersDict:
  speakers_path = get_speakers_json(base_dir, ds_name)
  return SpeakersDict.load(speakers_path)
  
def save_speaker_json(base_dir: str, ds_name: str, speakers: SpeakersDict):
  speakers_path = get_speakers_json(base_dir, ds_name)
  speakers.save(speakers_path)

def save_speaker_log_json(base_dir: str, ds_name: str, speakers_log: SpeakersLogDict):
  speakers_path = get_speakers_log_json(base_dir, ds_name)
  speakers_log.save(speakers_path)

def get_ds_csv(base_dir: str, ds_name: str):
  return os.path.join(get_ds_root_dir(base_dir, ds_name, True), 'data.csv')

def get_speakers_json(base_dir: str, ds_name: str):
  return os.path.join(get_ds_root_dir(base_dir, ds_name, True), 'speakers.json')

def get_speakers_log_json(base_dir: str, ds_name: str):
  return os.path.join(get_ds_root_dir(base_dir, ds_name, True), 'speakers_log.json')

#endregion

#region Wav

def _get_wav_root_dir(base_dir: str, ds_name: str, create: bool = False):
  return get_subdir(get_ds_root_dir(base_dir, ds_name), "wav", create)

def get_wav_subdir(base_dir: str, ds_name: str, sub_name: str, create: bool = False):
  return get_subdir(_get_wav_root_dir(base_dir, ds_name, create), sub_name, create)

def load_wav_csv(base_dir: str, ds_name: str, sub_name: str) -> WavDataList:
  origin_wav_data_path = get_wav_csv(base_dir, ds_name, sub_name)
  return WavDataList.load(origin_wav_data_path)
  
def save_wav_csv(base_dir: str, ds_name: str, sub_name: str, wav_data: WavDataList):
  wav_data_path = get_wav_csv(base_dir, ds_name, sub_name)
  wav_data.save(wav_data_path)

def wav_subdir_exists(base_dir: str, ds_name: str, sub_name: str):
  wav_data_dir = get_wav_subdir(base_dir, ds_name, sub_name, create=False)
  return os.path.exists(wav_data_dir)

def get_wav_csv(base_dir: str, ds_name: str, sub_name: str):
  subdir = get_wav_subdir(base_dir, ds_name, sub_name, create=True)
  return os.path.join(subdir, "data.csv")

#endregion

#region Mel

def _get_mel_root_dir(base_dir: str, ds_name: str, create: bool = False):
  return get_subdir(get_ds_root_dir(base_dir, ds_name), "mel", create)

def get_mel_subdir(base_dir: str, ds_name: str, sub_name: str, create: bool = False):
  return get_subdir(_get_mel_root_dir(base_dir, ds_name, create), sub_name, create)

def load_mel_csv(base_dir: str, ds_name: str, sub_name: str) -> MelDataList:
  data_path = get_mel_csv(base_dir, ds_name, sub_name)
  return MelDataList.load(data_path)
  
def save_mel_csv(base_dir: str, ds_name: str, sub_name: str, mel_data: MelDataList):
  data_path = get_mel_csv(base_dir, ds_name, sub_name)
  mel_data.save(data_path)

def mel_subdir_exists(base_dir: str, ds_name: str, sub_name: str):
  data_dir = get_mel_subdir(base_dir, ds_name, sub_name, create=False)
  return os.path.exists(data_dir)

def get_mel_csv(base_dir: str, ds_name: str, sub_name: str):
  subdir = get_mel_subdir(base_dir, ds_name, sub_name, create=True)
  return os.path.join(subdir, "data.csv")

#endregion

#region Text

def _get_text_root_dir(base_dir: str, ds_name: str, create: bool = False):
  return get_subdir(get_ds_root_dir(base_dir, ds_name), "text", create)

def get_text_subdir(base_dir: str, ds_name: str, sub_name: str, create: bool = False):
  return get_subdir(_get_text_root_dir(base_dir, ds_name, create), sub_name, create)

def load_text_symbol_converter(base_dir: str, ds_name: str, sub_name: str) -> SymbolConverter:
  data_path = get_text_symbol_converter(base_dir, ds_name, sub_name)
  return SymbolConverter.load_from_file(data_path)
  
def save_text_symbol_converter(base_dir: str, ds_name: str, sub_name: str, data: SymbolConverter):
  data_path = get_text_symbol_converter(base_dir, ds_name, sub_name)
  data.dump(data_path)

def load_text_symbols_json(base_dir: str, ds_name: str, sub_name: str) -> SymbolsDict:
  data_path = get_text_symbols_json(base_dir, ds_name, sub_name)
  return SymbolsDict.load(data_path)
  
def save_text_symbols_json(base_dir: str, ds_name: str, sub_name: str, data: SymbolsDict):
  data_path = get_text_symbols_json(base_dir, ds_name, sub_name)
  data.save(data_path)

def load_text_csv(base_dir: str, ds_name: str, sub_name: str) -> TextDataList:
  origin_data_path = get_text_csv(base_dir, ds_name, sub_name)
  return TextDataList.load(origin_data_path)
  
def save_text_csv(base_dir: str, ds_name: str, sub_name: str, data: TextDataList):
  data_path = get_text_csv(base_dir, ds_name, sub_name)
  data.save(data_path)

def text_subdir_exists(base_dir: str, ds_name: str, sub_name: str):
  data_dir = get_text_subdir(base_dir, ds_name, sub_name, create=False)
  return os.path.exists(data_dir)

def get_text_csv(base_dir: str, ds_name: str, sub_name: str):
  subdir = get_text_subdir(base_dir, ds_name, sub_name, create=True)
  return os.path.join(subdir, "data.csv")

def get_text_symbols_json(base_dir: str, ds_name: str, sub_name: str):
  subdir = get_text_subdir(base_dir, ds_name, sub_name, create=True)
  return os.path.join(subdir, "symbols.json")

def get_text_symbol_converter(base_dir: str, ds_name: str, sub_name: str):
  subdir = get_text_subdir(base_dir, ds_name, sub_name, create=True)
  return os.path.join(subdir, "symbol_ids.json")

#endregion
