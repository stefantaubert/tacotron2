import os

from src.app.pre.io import get_pre_dir
from src.core.common import get_subdir, SpeakersDict, SpeakersLogDict, SymbolIdDict, AccentsDict
from src.core.pre import (DsData, DsDataList,
                          PreparedDataList,
                          ljs_preprocess,
                          thchs_kaldi_preprocess, thchs_preprocess)
# don't do preprocessing here because inconsistent with mels because it is not always usefull to calc mels instand
# from src.app.pre.text import preprocess_text
# from src.app.pre.wav import preprocess_wavs
# from src.app.pre.mel import preprocess_mels

_ds_data_csv = "data.csv"
_ds_speakers_json = "speakers.json"
_ds_symbols_json = "symbols.json"
_ds_accents_json = "accents.json"


def _get_ds_root_dir(base_dir: str, create: bool = False):
  return get_subdir(get_pre_dir(base_dir, create), "ds", create)


def get_ds_dir(base_dir: str, ds_name: str, create: bool = False):
  return get_subdir(_get_ds_root_dir(base_dir, create), ds_name, create)


def load_ds_csv(ds_dir: str) -> DsDataList:
  path = os.path.join(ds_dir, _ds_data_csv)
  return DsDataList.load(DsData, path)


def _save_ds_csv(ds_dir: str, result: DsDataList):
  path = os.path.join(ds_dir, _ds_data_csv)
  result.save(path)


def load_symbols_json(ds_dir: str) -> SymbolIdDict:
  path = os.path.join(ds_dir, _ds_symbols_json)
  return SymbolIdDict.load_from_file(path)


def _save_symbols_json(ds_dir: str, data: SymbolIdDict):
  path = os.path.join(ds_dir, _ds_symbols_json)
  data.save(path)


def load_accents_json(ds_dir: str) -> AccentsDict:
  path = os.path.join(ds_dir, _ds_accents_json)
  return AccentsDict.load(path)


def _save_accents_json(ds_dir: str, data: AccentsDict):
  path = os.path.join(ds_dir, _ds_accents_json)
  data.save(path)


def load_speaker_json(ds_dir: str) -> SpeakersDict:
  path = os.path.join(ds_dir, _ds_speakers_json)
  return SpeakersDict.load(path)


def _save_speaker_json(ds_dir: str, speakers: SpeakersDict):
  path = os.path.join(ds_dir, _ds_speakers_json)
  speakers.save(path)


def _save_speaker_log_json(ds_dir: str, speakers_log: SpeakersLogDict):
  path = os.path.join(ds_dir, "speakers_log.json")
  speakers_log.save(path)


def preprocess_thchs(base_dir: str, ds_name: str, path: str, auto_dl: bool):
  print("Preprocessing THCHS-30 dataset...")
  _preprocess_ds(base_dir, ds_name, path, auto_dl, thchs_preprocess)


def preprocess_thchs_kaldi(base_dir: str, ds_name: str, path: str, auto_dl: bool):
  print("Preprocessing THCHS-30 (Kaldi-Version) dataset...")
  _preprocess_ds(base_dir, ds_name, path, auto_dl, thchs_kaldi_preprocess)


def preprocess_ljs(base_dir: str, ds_name: str, path: str, auto_dl: bool):
  print("Preprocessing LJSpeech dataset...")
  _preprocess_ds(base_dir, ds_name, path, auto_dl, ljs_preprocess)


def _preprocess_ds(base_dir: str, ds_name: str, path: str, auto_dl: bool, preprocess_func):
  ds_path = get_ds_dir(base_dir, ds_name, create=False)
  if os.path.isdir(ds_path):
    print("Dataset already processed.")
  else:
    os.makedirs(ds_path)
    print("Reading data...")
    speakers, speakers_log, symbols, accents, ds_data = preprocess_func(path, auto_dl)
    _save_speaker_json(ds_path, speakers)
    _save_speaker_log_json(ds_path, speakers_log)
    _save_symbols_json(ds_path, symbols)
    _save_accents_json(ds_path, accents)
    _save_ds_csv(ds_path, ds_data)
    print("Dataset processed.")


if __name__ == "__main__":
  preprocess_ljs(
    base_dir="/datasets/models/taco2pt_v5",
    path="/datasets/LJSpeech-1.1",
    ds_name="ljs",
    auto_dl=True,
  )

  preprocess_thchs(
    path="/datasets/thchs_wav",
    base_dir="/datasets/models/taco2pt_v5",
    ds_name="thchs",
    auto_dl=True,
  )

  preprocess_thchs_kaldi(
    base_dir="/datasets/models/taco2pt_v5",
    path="/datasets/THCHS-30",
    ds_name="thchs_kaldi",
    auto_dl=True,
  )
