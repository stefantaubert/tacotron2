import os

from src.app.pre.io import get_pre_dir
from src.core.common import get_subdir
from src.core.pre import (DsData, DsDataList,
                          PreparedDataList, SpeakersDict,
                          SpeakersLogDict,
                          ljs_preprocess,
                          thchs_kaldi_preprocess, thchs_preprocess)

_ds_data_csv = "data.csv"
_ds_speakers_json = "speakers.json"

def _get_ds_root_dir(base_dir: str, create: bool = False):
  return get_subdir(get_pre_dir(base_dir, create), "ds", create)

def get_ds_dir(base_dir: str, ds_name: str, create: bool = False):
  return get_subdir(_get_ds_root_dir(base_dir, create), ds_name, create)

def load_ds_csv(ds_dir: str) -> DsDataList:
  path = os.path.join(ds_dir, _ds_data_csv)
  return DsDataList.load(path)
  
def _save_ds_csv(ds_dir: str, result: DsDataList):
  path = os.path.join(ds_dir, _ds_data_csv)
  result.save(path)

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
  _preprocess_ds(base_dir, ds_name, path, auto_dl, thchs_preprocess)

def preprocess_thchs_kaldi(base_dir: str, ds_name: str, path: str, auto_dl: bool):
  _preprocess_ds(base_dir, ds_name, path, auto_dl, thchs_kaldi_preprocess)

def preprocess_ljs(base_dir: str, ds_name: str, path: str, auto_dl: bool):
  _preprocess_ds(base_dir, ds_name, path, auto_dl, ljs_preprocess)

def _preprocess_ds(base_dir: str, ds_name: str, path: str, auto_dl: bool, preprocess_func):
  ds_path = get_ds_dir(base_dir, ds_name, create=False)
  if os.path.isdir(ds_path):
    print("Dataset already processed.")
  else:
    os.makedirs(ds_path)
    print("Reading data...")
    speakers, speakers_log, ds_data = preprocess_func(path, auto_dl)
    _save_speaker_json(ds_path, speakers)
    _save_speaker_log_json(ds_path, speakers_log)
    _save_ds_csv(ds_path, ds_data)
    print("Dataset processed.")

if __name__ == "__main__":
  preprocess_ljs(
    base_dir="/datasets/models/taco2pt_v4",
    path="/datasets/LJSpeech-1.1",
    ds_name="ljs",
    auto_dl=True,
  )

  preprocess_thchs(
    path="/datasets/thchs_wav",
    base_dir="/datasets/models/taco2pt_v4",
    ds_name="thchs",
    auto_dl=True,
  )

  preprocess_thchs_kaldi(
    base_dir="/datasets/models/taco2pt_v4",
    path="/datasets/THCHS-30",
    ds_name="thchs_kaldi",
    auto_dl=True,
  )
