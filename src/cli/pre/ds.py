import os
from src.core.pre import DsDataList, SpeakersDict, SpeakersLogDict, ljs_preprocess, thchs_preprocess, thchs_kaldi_preprocess
from src.cli.pre.paths import get_ds_csv, get_speakers_json, get_ds_dir, get_speakers_log_json
from argparse import ArgumentParser

#region IO

def load_ds_csv(base_dir: str, ds_name: str) -> DsDataList:
  result_path = get_ds_csv(base_dir, ds_name)
  return DsDataList.load(result_path)
  
def _save_ds_csv(base_dir: str, ds_name: str, result: DsDataList):
  result_path = get_ds_csv(base_dir, ds_name)
  result.save(result_path)

def load_speaker_json(base_dir: str, ds_name: str) -> SpeakersDict:
  speakers_path = get_speakers_json(base_dir, ds_name)
  return SpeakersDict.load(speakers_path)
  
def _save_speaker_json(base_dir: str, ds_name: str, speakers: SpeakersDict):
  speakers_path = get_speakers_json(base_dir, ds_name)
  speakers.save(speakers_path)

def _save_speaker_log_json(base_dir: str, ds_name: str, speakers_log: SpeakersLogDict):
  speakers_path = get_speakers_log_json(base_dir, ds_name)
  speakers_log.save(speakers_path)

#endregion

def _preprocess(base_dir: str, ds_name: str, path: str, auto_dl: bool, preprocess_func):
  ds_path = get_ds_dir(base_dir, ds_name, create=False)
  if os.path.isdir(ds_path):
    print("Dataset already processed.")
  else:
    print("Reading data...")
    speakers, speakers_log, ds_data = preprocess_func(path, auto_dl)
    _save_speaker_json(base_dir, ds_name, speakers)
    _save_speaker_log_json(base_dir, ds_name, speakers_log)
    _save_ds_csv(base_dir, ds_name, ds_data)
    print("Dataset processed.")

#region THCHS

def init_thchs_parser(parser: ArgumentParser):
  parser.add_argument('--path', type=str, required=True, help='THCHS dataset directory')
  parser.add_argument('--auto_dl', action="store_true")
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--ds_name', type=str, required=True, default='thchs')
  return _read_thchs

def _read_thchs(base_dir: str, ds_name: str, path: str, auto_dl: bool):
  _preprocess(base_dir, ds_name, path, auto_dl, thchs_preprocess)

#endregion

#region THCHS (Kaldi)

def init_thchs_kaldi_parser(parser: ArgumentParser):
  parser.add_argument('--path', type=str, required=True, help='THCHS dataset directory')
  parser.add_argument('--auto_dl', action="store_true")
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--ds_name', type=str, required=True, default='thchs_kaldi')
  return _read_thchs_kaldi

def _read_thchs_kaldi(base_dir: str, ds_name: str, path: str, auto_dl: bool):
  _preprocess(base_dir, ds_name, path, auto_dl, thchs_kaldi_preprocess)

#endregion

#region LJSpeech

def init_ljs_parser(parser: ArgumentParser):
  parser.add_argument('--path', type=str, required=True, help='LJS dataset directory')
  parser.add_argument('--auto_dl', action="store_true")
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--ds_name', type=str, required=True, default='ljs')
  return _read_ljs

def _read_ljs(base_dir: str, ds_name: str, path: str, auto_dl: bool):
  _preprocess(base_dir, ds_name, path, auto_dl, ljs_preprocess)

#endregion

if __name__ == "__main__":
 
  _read_thchs(
    path="/datasets/thchs_wav",
    base_dir="/datasets/models/taco2pt_v2",
    ds_name="thchs",
    auto_dl=True,
  )

  _read_ljs(
    base_dir="/datasets/models/taco2pt_v2",
    path="/datasets/LJSpeech-1.1",
    ds_name="ljs",
    auto_dl=True,
  )

  # __read_thchs_kaldi(
  #   base_dir="/datasets/models/taco2pt_v2",
  #   path="/datasets/THCHS-30",
  #   ds_name="thchs_kaldi",
  #   auto_dl=True,
  # )
