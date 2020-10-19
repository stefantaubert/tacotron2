import os
from typing import List, Set, Tuple

from src.app.pre.ds import (get_ds_dir, load_accents_json, load_ds_csv,
                            load_speaker_json)
from src.app.pre.io import get_pre_dir
from src.app.pre.mel import get_mel_dir, load_mel_csv
from src.app.pre.text import (get_text_dir, load_text_csv,
                              load_text_symbol_converter)
from src.app.pre.wav import get_wav_dir, load_wav_csv
from src.core.common.accents_dict import AccentsDict
from src.core.common.speakers_dict import SpeakersDict
from src.core.common.symbol_id_dict import SymbolIdDict
from src.core.common.utils import get_subdir
from src.core.pre.ds import DsDataList
from src.core.pre.mel import MelDataList
from src.core.pre.merge_ds import (DsDataset, DsDatasetList, PreparedData,
                                   PreparedDataList, preprocess)
from src.core.pre.text.pre import TextDataList
from src.core.pre.wav import WavDataList


def _get_prepared_root_dir(base_dir: str, create: bool = False):
  return get_subdir(get_pre_dir(base_dir, create), 'prepared', create)


_prepared_data_csv = "data.csv"
_prepared_speakers_json = "speakers.json"
_prepared_symbols_json = "symbols.json"
_prepared_accents_json = "accents.json"


def get_prepared_dir(base_dir: str, prep_name: str, create: bool = False):
  return get_subdir(_get_prepared_root_dir(base_dir, create), prep_name, create)


def load_filelist(prep_dir: str) -> PreparedDataList:
  path = os.path.join(prep_dir, _prepared_data_csv)
  return PreparedDataList.load(PreparedData, path)


def save_filelist(prep_dir: str, result: PreparedDataList):
  path = os.path.join(prep_dir, _prepared_data_csv)
  result.save(path)


def load_prep_speakers_json(prep_dir: str) -> SpeakersDict:
  path = os.path.join(prep_dir, _prepared_speakers_json)
  return SpeakersDict.load(path)


def save_prep_speakers_json(prep_dir: str, speakers: SpeakersDict):
  path = os.path.join(prep_dir, _prepared_speakers_json)
  speakers.save(path)


def load_prep_symbol_converter(prep_dir: str) -> SymbolIdDict:
  path = os.path.join(prep_dir, _prepared_symbols_json)
  return SymbolIdDict.load_from_file(path)


def save_prep_symbol_converter(prep_dir: str, data: SymbolIdDict):
  path = os.path.join(prep_dir, _prepared_symbols_json)
  data.save(path)


def load_prep_accents_ids(prep_dir: str) -> AccentsDict:
  path = os.path.join(prep_dir, _prepared_accents_json)
  return AccentsDict.load(path)


def save_prep_accents_ids(prep_dir: str, data: AccentsDict):
  path = os.path.join(prep_dir, _prepared_accents_json)
  data.save(path)


def prepare_ds(base_dir: str, prep_name: str, ds_speakers: List[Tuple[str, str]], ds_text_audio: List[Tuple[str, str, str]]):
  print(f"Preparing dataset: {prep_name}...")
  prep_dir = get_prepared_dir(base_dir, prep_name)
  if os.path.isdir(prep_dir):
    print("Already created.")
  else:
    datasets = DsDatasetList()
    for ds_name, text_name, audio_name in ds_text_audio:
      # multiple uses of one ds are not valid

      ds_dir = get_ds_dir(base_dir, ds_name)
      text_dir = get_text_dir(ds_dir, text_name)
      wav_dir = get_wav_dir(ds_dir, audio_name)
      mel_dir = get_mel_dir(ds_dir, audio_name)

      ds_dataset = DsDataset(
        name=ds_name,
        data=load_ds_csv(ds_dir),
        texts=load_text_csv(text_dir),
        wavs=load_wav_csv(wav_dir),
        mels=load_mel_csv(mel_dir),
        speakers=load_speaker_json(ds_dir),
        symbol_ids=load_text_symbol_converter(text_dir),
        accent_ids=load_accents_json(ds_dir)
      )

      datasets.append(ds_dataset)

    merged_data = preprocess(
      datasets=datasets,
      ds_speakers=ds_speakers
    )

    prep_data = PreparedDataList.init_from_merged_ds(merged_data.data)

    os.makedirs(prep_dir)
    save_filelist(prep_dir, prep_data)
    save_prep_symbol_converter(prep_dir, merged_data.symbol_ids)
    save_prep_accents_ids(prep_dir, merged_data.accent_ids)
    save_prep_speakers_json(prep_dir, merged_data.speaker_ids)


if __name__ == "__main__":
  if False:
    prepare_ds(
      base_dir="/datasets/models/taco2pt_v5",
      prep_name="debug",
      ds_speakers=[("ljs", "all"), ("thchs", "all"), ("thchs", "B4")],
      #ds_text_audio=[("thchs", "ipa", "22050Hz_normalized_nosil")]
      ds_text_audio=[("ljs", "ipa_norm", "22050kHz"), ("thchs", "ipa", "22050Hz_normalized_nosil")]
    )
  else:
    prep_dir = get_prepared_dir(
      base_dir="/datasets/models/taco2pt_v5",
      prep_name="debug",
    )

    res = load_filelist(
      prep_dir=prep_dir,
    )
    

  # prepare_ds(
  #   base_dir="/datasets/models/taco2pt_v5",
  #   prep_name="thchs_ljs_ipa",
  #   ds_speakers=[("ljs", "all"), ("thchs", "all")],
  #   ds_text_audio=[("ljs", "ipa_norm", "22050Hz"), ("thchs", "ipa", "22050kHz_normalized_nosil")]
  # )

  # prepare_ds(
  #   base_dir="/datasets/models/taco2pt_v5",
  #   prep_name="ljs",
  #   ds_speakers=[("ljs", "all")],
  #   ds_text_audio=[("ljs", "ipa_norm", "22050Hz")]
  # )

  # prepare_ds(
  #   base_dir="/datasets/models/taco2pt_v5",
  #   prep_name="thchs",
  #   ds_speakers=[("thchs", "all")],
  #   ds_text_audio=[("thchs", "ipa", "22050Hz_norm_wo_sil")]
  # )
