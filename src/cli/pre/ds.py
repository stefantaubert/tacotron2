import os
from src.core.pre.ds import get_all_speakers, get_ds_data
from src.core.pre.parser import PreDataList
from src.core.pre.ds import DsDataList, SpeakersDict, SpeakersLogDict
from src.cli.pre.paths import get_ds_csv, get_speakers_json, get_ds_dir, get_speakers_log_json
from argparse import ArgumentParser

#region IO

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

#endregion

def process(base_dir: str, ds_name: str, data: PreDataList):
  print("Reading data...")

  speakers, speakers_log = get_all_speakers(data)
  save_speaker_json(base_dir, ds_name, speakers)
  save_speaker_log_json(base_dir, ds_name, speakers_log)

  result = get_ds_data(data, speakers)
  save_ds_csv(base_dir, ds_name, result)

  print("Dataset processed.")

def __read_wavs_ds(base_dir: str, ds_name: str, path: str, auto_dl: bool, parse, dl):
  ds_path = get_ds_dir(base_dir, ds_name, create=False)
  if os.path.isdir(ds_path):
    print("Dataset already processed.")
  else:
    if auto_dl:
      dl(path)
    data = parse(path)
    process(base_dir, ds_name, data)

#region THCHS

def init_thchs_parser(parser: ArgumentParser):
  parser.add_argument('--path', type=str, required=True, help='THCHS dataset directory')
  parser.add_argument('--auto_dl', action="store_true")
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--ds_name', type=str, required=True, default='thchs')
  return __read_wavs_thchs

def __read_wavs_thchs(base_dir: str, ds_name: str, path: str, auto_dl: bool):
  from src.core.pre.parser.thchs import parse, ensure_downloaded
  __read_wavs_ds(base_dir, ds_name, path, auto_dl, parse, ensure_downloaded)

#endregion

#region THCHS (Kaldi)

def init_thchs_kaldi_parser(parser: ArgumentParser):
  parser.add_argument('--path', type=str, required=True, help='THCHS dataset directory')
  parser.add_argument('--auto_dl', action="store_true")
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--ds_name', type=str, required=True, default='thchs_kaldi')
  return __read_wavs_thchs_kaldi

def __read_wavs_thchs_kaldi(base_dir: str, ds_name: str, path: str, auto_dl: bool):
  from src.core.pre.parser.thchs_kaldi import parse, ensure_downloaded
  __read_wavs_ds(base_dir, ds_name, path, auto_dl, parse, ensure_downloaded)

#endregion

#region LJSpeech

def init_ljs_parser(parser: ArgumentParser):
  parser.add_argument('--path', type=str, required=True, help='LJS dataset directory')
  parser.add_argument('--auto_dl', action="store_true")
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--ds_name', type=str, required=True, default='ljs')
  return __read_wavs_ljs

def __read_wavs_ljs(base_dir: str, ds_name: str, path: str, auto_dl: bool):
  from src.core.pre.parser.ljs import parse, ensure_downloaded
  __read_wavs_ds(base_dir, ds_name, path, auto_dl, parse, ensure_downloaded)

#endregion

if __name__ == "__main__":
 
  __read_wavs_thchs(
    path="/datasets/thchs_wav",
    base_dir="/datasets/models/taco2pt_v2",
    ds_name="thchs",
    auto_dl=True,
  )

  __read_wavs_ljs(
    base_dir="/datasets/models/taco2pt_v2",
    path="/datasets/LJSpeech-1.1",
    ds_name="ljs",
    auto_dl=True,
  )

  # __read_wavs_thchs_kaldi(
  #   base_dir="/datasets/models/taco2pt_v2",
  #   path="/datasets/THCHS-30",
  #   ds_name="thchs_kaldi",
  #   auto_dl=True,
  # )
