import os
from src.core.pre.ds.data import DsData, DsDataList, SpeakersDict
from src.core.pre.parser.data import PreData, PreDataList
from src.cli.pre.paths import get_pre_ds_data_file, get_pre_ds_speakers_file, get_pre_ds_path


def process(base_dir: str, ds_name: str, data: PreDataList):
  print("Reading data...")

  speakers_path = get_pre_ds_speakers_file(base_dir, ds_name, create=True)
  speakers = SpeakersDict.fromlist(data)
  speakers.save(speakers_path)

  result_path = get_pre_ds_data_file(base_dir, ds_name, create=True)
  result = DsDataList.fromlist(data, speakers)
  result.save(result_path)

  print("Dataset processed.")

def __read_wavs_ds(base_dir: str, ds_name: str, path: str, parse):
  ds_path = get_pre_ds_path(base_dir, ds_name, create=False)
  if os.path.isdir(ds_path):
    print("Dataset already processed.")
  else:
    data = parse(path)
    process(base_dir, ds_name, data)

def __read_wavs_thchs(base_dir: str, ds_name: str, path: str, auto_dl: bool):
  from src.core.pre.parser.thchs import parse, ensure_downloaded
  if auto_dl:
    ensure_downloaded(path)
  __read_wavs_ds(base_dir, ds_name, path, parse)
  

if __name__ == "__main__":
 
  __read_wavs_thchs(
    path="/datasets/thchs_wav",
    base_dir="/datasets/models/taco2pt_v2",
    ds_name="thchs",
    auto_dl=True,
  )

  # __read_wavs_ljs(
  #   base_dir="/datasets/models/taco2pt_v2",
  #   path="/datasets/LJSpeech-1.1",
  #   name="ljs_22050kHz",
  #   auto_dl=True,
  # )

  # __read_wavs_thchs_kaldi(
  #   base_dir="/datasets/models/taco2pt_v2",
  #   path="/datasets/THCHS-30",
  #   name="thchs_kaldi_16000kHz",
  #   auto_dl=True,
  # )
