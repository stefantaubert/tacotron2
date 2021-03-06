import random
from dataclasses import dataclass
from typing import Dict, List, Optional, OrderedDict, Set, Tuple

from src.core.common.accents_dict import AccentsDict
from src.core.common.gender import Gender
from src.core.common.language import Language
from src.core.common.speakers_dict import SpeakersDict
from src.core.common.symbol_id_dict import SymbolIdDict
from src.core.common.text import deserialize_list
from src.core.common.train import split_train_test_val
from src.core.common.utils import GenericList
from src.core.pre.ds import DsDataList
from src.core.pre.mel import MelDataList
from src.core.pre.text.pre import TextDataList
from src.core.pre.wav import WavDataList

ALL_SPEAKERS_INDICATOR = "all"


@dataclass
class DsDataset():
  name: str
  data: DsDataList
  texts: TextDataList
  wavs: WavDataList
  mels: MelDataList
  speakers: SpeakersDict
  symbol_ids: SymbolIdDict
  accent_ids: AccentsDict


@dataclass
class DsDatasetList(GenericList[DsDataset]):
  pass


@dataclass
class MergedDatasetEntry():
  entry_id: int
  basename: str
  speaker_id: int
  serialized_symbol_ids: str
  serialized_accent_ids: str
  gender: Gender
  lang: Language
  wav_path: str
  duration: float
  sampling_rate: int
  mel_path: str
  n_mel_channels: int

  def load_init(self):
    self.lang = Language(self.lang)
    self.gender = Gender(self.gender)


@dataclass
class MergedDataset(GenericList[MergedDatasetEntry]):
  def load_init(self):
    for item in self.items():
      item.load_init()

  @classmethod
  def init_from_ds_dataset(cls, ds: DsDataset):
    res = cls()
    for data in zip(ds.data.items(), ds.texts.items(), ds.wavs.items(), ds.mels.items()):
      entry, text, wav, mel = data
      new_entry = MergedDatasetEntry(
        entry_id=entry.entry_id,
        gender=entry.gender,
        basename=entry.basename,
        speaker_id=entry.speaker_id,
        lang=text.lang,
        serialized_accent_ids=text.serialized_accent_ids,
        serialized_symbol_ids=text.serialized_symbol_ids,
        wav_path=wav.wav,
        duration=wav.duration,
        sampling_rate=wav.sr,
        mel_path=mel.mel_path,
        n_mel_channels=mel.n_mel_channels,
      )
      res.append(new_entry)
    return res


class MergedDatasetContainer():
  def __init__(self, data: MergedDataset, name: Optional[str], speaker_ids: SpeakersDict, symbol_ids: SymbolIdDict, accent_ids: AccentsDict) -> None:
    self.data = data
    self.name = name
    self.symbol_ids = symbol_ids
    self.accent_ids = accent_ids
    self.speaker_ids = speaker_ids

  @classmethod
  def init_from_ds_dataset(cls, ds: DsDataset):
    data = MergedDataset.init_from_ds_dataset(ds)
    res = cls(
      data=data,
      name=ds.name,
      speaker_ids=ds.speakers,
      symbol_ids=ds.symbol_ids,
      accent_ids=ds.accent_ids,
    )
    return res

  def remove_speakers(self, speakers: Set[int]) -> None:
    res = MergedDataset()
    for entry in self.data.items():
      if entry.speaker_id not in speakers:
        res.append(entry)
    self.data = res
    self._remove_unused_speakers()
    self._remove_unused_symbols()
    self._remove_unused_accents()

  def _remove_unused_symbols(self) -> None:
    all_symbol_ids: Set[int] = set()
    for entry in self.data.items():
      all_symbol_ids |= set(deserialize_list(entry.serialized_symbol_ids))
    unused_symbol_ids = self.symbol_ids.get_all_symbol_ids().difference(all_symbol_ids)
    # unused_symbols = unused_symbols.difference({PADDING_SYMBOL})
    self.symbol_ids.remove_ids(unused_symbol_ids)

  def _remove_unused_accents(self) -> None:
    all_accent_ids: Set[int] = set()
    for entry in self.data.items():
      all_accent_ids |= set(deserialize_list(entry.serialized_accent_ids))
    unused_accent_ids = self.accent_ids.get_all_ids().difference(all_accent_ids)
    # unused_symbols = unused_symbols.difference({PADDING_SYMBOL})
    self.accent_ids.remove_ids(unused_accent_ids)

  def _remove_unused_speakers(self) -> None:
    all_speaker_ids: Set[int] = set()
    for entry in self.data.items():
      all_speaker_ids |= {entry.speaker_id}
    unused_speaker_ids = set(self.speaker_ids.get_all_speaker_ids()).difference(all_speaker_ids)
    # unused_symbols = unused_symbols.difference({PADDING_SYMBOL})
    self.speaker_ids.remove_ids(unused_speaker_ids)


class MergedDatasetContainerList():
  def __init__(self, data: List[MergedDatasetContainer]) -> None:
    self.data = data

  def merge(self) -> MergedDatasetContainer:
    accent_ids = self.make_common_accent_ids()
    speaker_ids = self.make_common_speaker_ids()
    symbol_ids = self.make_common_symbol_ids()

    new_ds = MergedDataset()
    for dataset in self.data:
      for entry in dataset.data.items():
        new_ds.append(entry)

    # TODO: maybe sorting after speakerid and then entry_id

    # # Set new entry_id
    # for i, entry in enumerate(new_ds.items()):
    #   entry.entry_id = i

    res = MergedDatasetContainer(
      data=new_ds,
      name=None,
      accent_ids=accent_ids,
      speaker_ids=speaker_ids,
      symbol_ids=symbol_ids,
    )

    return res

  def make_common_symbol_ids(self) -> SymbolIdDict:
    all_symbols: Set[str] = set()
    for ds in self.data:
      all_symbols |= ds.symbol_ids.get_all_symbols()
    new_symbol_ids = SymbolIdDict.init_from_symbols_with_pad(all_symbols)

    for ds in self.data:
      for entry in ds.data.items():
        original_symbols = ds.symbol_ids.get_symbols(entry.serialized_symbol_ids)
        entry.serialized_symbol_ids = new_symbol_ids.get_serialized_ids(original_symbols)
      ds.symbol_ids = new_symbol_ids

    return new_symbol_ids

  def make_common_accent_ids(self) -> AccentsDict:
    all_accents: Set[str] = set()
    for ds in self.data:
      all_accents |= ds.accent_ids.get_all_accents()
    new_accent_ids = AccentsDict.init_from_accents_with_pad(all_accents)

    for ds in self.data:
      for entry in ds.data.items():
        original_accents = ds.accent_ids.get_accents(entry.serialized_accent_ids)
        entry.serialized_accent_ids = new_accent_ids.get_serialized_ids(original_accents)
      ds.accent_ids = new_accent_ids

    return new_accent_ids

  @staticmethod
  def get_new_speaker_name(ds_name: str, speaker_name: str) -> str:
    return f"{ds_name},{speaker_name}"

  def make_common_speaker_ids(self) -> SpeakersDict:
    all_speakers: List[str] = list()
    for ds in self.data:
      old_speaker_names = ds.speaker_ids.get_all_speakers()
      new_speaker_names = [MergedDatasetContainerList.get_new_speaker_name(
        ds.name, old_name) for old_name in old_speaker_names]
      all_speakers.extend(new_speaker_names)

    all_speakers_have_unique_names = len(all_speakers) == len(set(all_speakers))
    assert all_speakers_have_unique_names

    new_speaker_ids = SpeakersDict.fromlist(all_speakers)

    for ds in self.data:
      for entry in ds.data.items():
        old_speaker_name = ds.speaker_ids.get_speaker(entry.speaker_id)
        new_speaker_name = MergedDatasetContainerList.get_new_speaker_name(
          ds.name, old_speaker_name)
        entry.speaker_id = new_speaker_ids.get_id(new_speaker_name)
      ds.speaker_ids = new_speaker_ids

    return new_speaker_ids


@dataclass()
class PreparedData:
  entry_id: int
  ds_entry_id: int
  wav_path: str
  mel_path: str
  serialized_symbol_ids: str
  serialized_accent_ids: str
  duration: float
  speaker_id: int

  def load_init(self):
    pass
    # self.duration = float(self.duration)
    # self.speaker_id = int(self.speaker_id)
    # self.entry_id = int(self.entry_id)
    # self.ds_entry_id = int(self.ds_entry_id)


class PreparedDataList(GenericList[PreparedData]):
  def load_init(self):
    for item in self.items():
      item.load_init()

  def get_total_duration_s(self):
    durations = [x.duration for x in self.items()]
    total_duration = sum(durations)
    return total_duration

  def get_for_validation(self, entry_id: Optional[int], speaker_id: Optional[int]) -> PreparedData:
    if entry_id is not None:
      return self._get_entry(entry_id)

    if speaker_id is not None:
      return self._get_random_entry_speaker_id(speaker_id)

    return self.get_random_entry()

  def _get_entry(self, entry_id: int) -> PreparedData:
    for entry in self.items():
      if entry.entry_id == entry_id:
        return entry
    raise Exception()

  def _get_random_entry_speaker_id(self, speaker_id: int) -> PreparedData:
    relevant_entries = [x for x in self.items() if x.speaker_id == speaker_id]
    assert len(relevant_entries) > 0
    entry = random.choice(relevant_entries)
    return entry

  @staticmethod
  def _get_key_for_sorting(elem: PreparedData) -> int:
    return elem.speaker_id, elem.ds_entry_id

  def custom_sort(self):
    self.sort(key=PreparedDataList._get_key_for_sorting, reverse=False)

  @classmethod
  def init_from_merged_ds(cls, merged_ds: MergedDataset):
    res = cls()
    for entry in merged_ds.items():
      prep_data = PreparedData(
        entry_id=-1,
        ds_entry_id=entry.entry_id,
        duration=entry.duration,
        mel_path=entry.mel_path,
        speaker_id=entry.speaker_id,
        serialized_accent_ids=entry.serialized_accent_ids,
        serialized_symbol_ids=entry.serialized_symbol_ids,
        wav_path=entry.wav_path,
      )
      res.append(prep_data)
    res.custom_sort()
    for i, entry in enumerate(res.items()):
      entry.entry_id = i
    return res


def _get_ds_speaker_ids(datasets: DsDatasetList, ds_speakers: List[Tuple[str, str]]) -> Dict[str, Set[int]]:
  speakers_dict = {ds.name: ds.speakers.get_all_speakers() for ds in datasets.items()}
  expanded_ds_speakers = expand_speakers(speakers_dict, ds_speakers)

  result: Dict[str, Set[int]] = dict()
  for ds_name, speaker_name in expanded_ds_speakers:
    for ds in datasets.items():
      if ds.name == ds_name:
        ds_speaker_id = ds.speakers.get_id(speaker_name)
        if ds_name not in result:
          result[ds_name] = set()
        result[ds_name] |= {ds_speaker_id}
        break

  return result


def preprocess(datasets: DsDatasetList, ds_speakers: List[Tuple[str, str]]) -> MergedDatasetContainer:
  ds_sepaker_ids = _get_ds_speaker_ids(datasets, ds_speakers)

  merged_datasets: List[MergedDatasetContainer] = list()
  for ds in datasets.items():
    if ds.name in ds_sepaker_ids.keys():
      merged_ds_container = MergedDatasetContainer.init_from_ds_dataset(ds)
      merged_datasets.append(merged_ds_container)

  for ds in merged_datasets:
    not_included_speakers = set(ds.speaker_ids.get_all_speaker_ids(
      )).difference(ds_sepaker_ids[ds.name])
    ds.remove_speakers(not_included_speakers)

  merged_dataset_container_list = MergedDatasetContainerList(
    data=merged_datasets
  )

  result = merged_dataset_container_list.merge()

  return result


def split_prepared_data_train_test_val(prep: PreparedDataList, test_size: float,
                                       validation_size: float, seed: int,
                                       shuffle: bool) -> Tuple[PreparedDataList,
                                                               PreparedDataList, PreparedDataList]:
  speaker_data: Dict[int, List[PreparedData]] = {}
  for data in prep.items():
    if data.speaker_id not in speaker_data:
      speaker_data[data.speaker_id] = []
    speaker_data[data.speaker_id].append(data)

  train = PreparedDataList()
  test = PreparedDataList()
  val = PreparedDataList()

  for _, data in speaker_data.items():
    speaker_train, speaker_test, speaker_val = split_train_test_val(
      wholeset=data,
      test_size=test_size,
      validation_size=validation_size,
      seed=seed, shuffle=shuffle,
      verbose=False
    )

    train.extend(speaker_train)
    test.extend(speaker_test)
    val.extend(speaker_val)

  return train, test, val


def expand_speakers(speakers_dict: OrderedDict[str, List[str]], ds_speakers: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
  # expand all
  expanded_speakers: List[Tuple[str, str]] = []
  for ds_name, speaker_name in ds_speakers:
    if ds_name not in speakers_dict:
      continue
    if speaker_name == ALL_SPEAKERS_INDICATOR:
      expanded_speakers.extend([(ds_name, speaker) for speaker in speakers_dict[ds_name]])
    else:
      if speaker_name not in speakers_dict[ds_name]:
        continue
      expanded_speakers.append((ds_name, speaker_name))
  expanded_speakers = list(sorted(set(expanded_speakers)))
  return expanded_speakers
