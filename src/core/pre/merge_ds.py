import random
from dataclasses import dataclass
from typing import List, OrderedDict, Set, Tuple

from sklearn.model_selection import train_test_split

from src.core.common.accents_dict import AccentsDict
from src.core.common.language import Language
from src.core.common.speakers_dict import SpeakersDict
from src.core.common.symbol_id_dict import SymbolIdDict
from src.core.common.utils import GenericList, get_unique_items
from src.core.pre.ds import DsData, DsDataList
from src.core.pre.mel import MelData, MelDataList
from src.core.pre.text import TextData, TextDataList
from src.core.pre.wav import WavData, WavDataList


@dataclass()
class PreparedData:
  i: int
  entry_id: int
  basename: str
  wav_path: str
  mel_path: str
  n_mel_channels: int
  serialized_symbol_ids: str
  serialized_accent_ids: str
  duration: float
  speaker_id: int
  speaker_name: str
  lang: Language  # can be removed
  ds_name: str

  def get_speaker_name(self):
    return str(self.speaker_name)


class PreparedDataList(GenericList[PreparedData]):
  def get_total_duration_s(self):
    durations = [x.duration for x in self.items()]
    total_duration = sum(durations)
    return total_duration

  def get_entry(self, i: int) -> PreparedData:
    for entry in self.items():
      if entry.i == i:
        return entry
    assert False

  def get_random_entry_ds_speaker(self, speaker_id: int) -> PreparedData:
    relevant_entries = [x for x in self.items() if x.speaker_id == speaker_id]
    assert len(relevant_entries) > 0
    entry = random.choice(relevant_entries)
    return entry

  @staticmethod
  def get_key_for_sorting_after_entry_id(elem: PreparedData) -> int:
    return elem.entry_id

  def sort_after_entry_id(self):
    self.sort(key=PreparedDataList.get_key_for_sorting_after_entry_id, reverse=False)


def preprocess(datasets: OrderedDict[str, Tuple[DsDataList, TextDataList, WavDataList, MelDataList, List[str], SymbolIdDict, AccentsDict]], ds_speakers: List[Tuple[str, str]]) -> Tuple[PreparedDataList, SymbolIdDict, AccentsDict, SpeakersDict]:
  speakers_dict = {k: v[4] for k, v in datasets.items()}
  expanded_ds_speakers = expand_speakers(speakers_dict, ds_speakers)
  ds_speakers_list, speakers_id_dict = get_speakers(expanded_ds_speakers)
  ds_prepared_data: List[Tuple[PreparedData, SymbolIdDict, AccentsDict]] = []

  for ds_name, dataset in datasets.items():
    ds_data, text_data, wav_data, mel_data, _, conv, accents = dataset
    speaker_names = ds_speakers_list[ds_name]
    prep = get_prepared_data(ds_name, ds_data, speaker_names, text_data, wav_data, mel_data)
    ds_prepared_data.append((prep, conv, accents))

  all_symbols = get_unique_items([conv.get_all_symbols() for _, conv, _ in ds_prepared_data])
  final_conv = SymbolIdDict.init_from_symbols_with_pad(all_symbols)
  all_accents = get_unique_items([accents.get_all_accents() for _, _, accents in ds_prepared_data])
  final_accents = AccentsDict.init_from_accents_with_pad(all_accents)
  whole = merge_prepared_data(ds_prepared_data, final_conv, final_accents)
  return whole, final_conv, final_accents, speakers_id_dict


def map_to_prepared_data(ds_name: str, ds_data: DsData, text_data: TextData,
                         wav_data: WavData, mel_data: MelData) -> PreparedData:
  prep_data = PreparedData(
    i=0,
    entry_id=ds_data.entry_id,
    basename=ds_data.basename,
    wav_path=wav_data.wav,
    mel_path=mel_data.mel_path,
    n_mel_channels=mel_data.n_mel_channels,
    serialized_symbol_ids=text_data.serialized_symbol_ids,
    serialized_accent_ids=text_data.serialized_accent_ids,
    lang=text_data.lang,
    duration=wav_data.duration,
    speaker_id=ds_data.speaker_id,
    speaker_name=ds_data.get_speaker_name(),
    ds_name=ds_name
  )

  return prep_data


def get_prepared_data(ds_name: str, ds_data: DsDataList, speaker_names: List[Tuple[str, int]], text_list: TextDataList, wav_list: WavDataList, mel_list: MelDataList) -> PreparedDataList:
  res = PreparedDataList()
  new_index = 0
  for speaker_name, new_speaker_id in speaker_names:
    for ds_entry in ds_data.items():
      if ds_entry.get_speaker_name() == speaker_name:
        prep_data = map_to_prepared_data(
          ds_name=ds_name,
          ds_data=ds_entry,
          text_data=text_list[ds_entry.entry_id],
          wav_data=wav_list[ds_entry.entry_id],
          mel_data=mel_list[ds_entry.entry_id]
        )

        prep_data.speaker_id = new_speaker_id
        prep_data.i = new_index
        new_index += 1

        res.append(prep_data)

  res.sort_after_entry_id()
  return res


def get_all_symbols(converters: List[SymbolIdDict]) -> Set[str]:
  all_symbols = set()
  for conv in converters:
    all_symbols = all_symbols.union(set(conv.get_all_symbols()))
  return all_symbols


def merge_prepared_data(prep_list: List[Tuple[PreparedDataList, SymbolIdDict, AccentsDict]],
                        new_symbol_ids: SymbolIdDict,
                        new_accent_ids: AccentsDict) -> PreparedDataList:
  res = PreparedDataList()
  new_index = 0
  for prep_data_list, old_symbol_ids, old_accent_ids in prep_list:
    for entry in prep_data_list.items():
      original_symbols = old_symbol_ids.get_symbols(entry.serialized_symbol_ids)
      entry.serialized_symbol_ids = new_symbol_ids.get_serialized_ids(original_symbols)
      original_accents = old_accent_ids.get_accents(entry.serialized_accent_ids)
      entry.serialized_accent_ids = new_accent_ids.get_serialized_ids(original_accents)
      entry.i = new_index
      new_index += 1
      res.append(entry)

  return res


def split_prepared_data_train_test_val(prep: PreparedDataList, test_size: float,
                                       val_size: float, seed: int,
                                       shuffle: bool) -> Tuple[PreparedDataList,
                                                               PreparedDataList, PreparedDataList]:
  speaker_data = {}
  for data in prep.items():
    if data.speaker_id not in speaker_data:
      speaker_data[data.speaker_id] = []
    speaker_data[data.speaker_id].append(data)

  train, test, val = [], [], []
  for _, data in speaker_data.items():
    speaker_train, speaker_test, speaker_val = split_train_test_val(
      data, test_size, val_size, seed, shuffle=shuffle)
    train.extend(speaker_train)
    test.extend(speaker_test)
    val.extend(speaker_val)

  return PreparedDataList(train), PreparedDataList(test), PreparedDataList(val)


def split_train_test_val(wholeset: list, test_size: float, validation_size: float, seed: int, shuffle: bool) -> Tuple[List, List, List]:
  assert seed >= 0
  assert 0 <= test_size <= 1
  assert 0 <= validation_size <= 1
  assert test_size + validation_size < 1

  trainset, testset, valset = wholeset, [], []

  if validation_size:
    is_ok = assert_fraction_is_big_enough(validation_size, len(trainset))
    trainset, valset = train_test_split(
      trainset, test_size=validation_size, random_state=seed, shuffle=shuffle)
    if not is_ok:
      check_is_not_empty(trainset)
      check_is_not_empty(valset)
      print(f"Split was however successfull, trainsize {len(trainset)}, valsize: {len(valset)}.")
  if test_size:
    adj_test_size = test_size / (1 - validation_size)
    is_ok = assert_fraction_is_big_enough(adj_test_size, len(trainset))
    trainset, testset = train_test_split(
      trainset, test_size=adj_test_size, random_state=seed, shuffle=shuffle)
    if not is_ok:
      check_is_not_empty(trainset)
      check_is_not_empty(valset)
      print(f"Split was however successfull, trainsize {len(trainset)}, testsize: {len(testset)}.")

  return trainset, testset, valset


def check_is_not_empty(dataset: PreparedDataList):
  if len(dataset) == 0:
    raise Exception("Aborting splitting, as a size of 0 resulted.")


def assert_fraction_is_big_enough(fraction: float, size: int):
  """tests that the fraction is bigger than the smallest fraction possible with that size to get at least one example in splitting"""
  calculation_inaccuracy = 10e-5
  min_frac = 1 / size
  y = min(fraction, 1 - fraction)
  if y + calculation_inaccuracy < min_frac:
    print(f"Warn: Split-fraction {fraction} is to small, it should be >= {min_frac}.")
    return False
  return True


def get_speakers(ds_speakers: Tuple[str, str]) -> Tuple[OrderedDict[str, List[Tuple[str, int]]], SpeakersDict]:
  """ Example:
  res = {
    "ljs": [("A12", 0), ("A13", 1)],
    "thchs": [("B12", 2)],
  }
  """
  res = OrderedDict()
  counter = 0
  speakers_dict = SpeakersDict()
  ds_speakers.sort()
  for ds_name, speaker_name in ds_speakers:
    if ds_name not in res:
      res[ds_name] = []
    res[ds_name].append((speaker_name, counter))
    speakers_dict[f"{ds_name},{speaker_name}"] = counter
    counter += 1

  return res, speakers_dict


def expand_speakers(speakers_dict: OrderedDict[str, List[str]], ds_speakers: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
  # expand all
  expanded_speakers: List[Tuple[str, str]] = []
  for ds_name, speaker_name in ds_speakers:
    if ds_name not in speakers_dict:
      continue
    if speaker_name == 'all':
      expanded_speakers.extend([(ds_name, speaker) for speaker in speakers_dict[ds_name]])
    else:
      if speaker_name not in speakers_dict[ds_name]:
        continue
      expanded_speakers.append((ds_name, speaker_name))
  expanded_speakers = list(sorted(set(expanded_speakers)))
  return expanded_speakers
