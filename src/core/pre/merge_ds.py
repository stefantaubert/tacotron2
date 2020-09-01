import random
from dataclasses import dataclass
from typing import List, OrderedDict, Tuple, Set

from sklearn.model_selection import train_test_split

from src.core.common import load_csv, parse_json, SpeakersDict, save_csv, save_json, SymbolIdDict, Language
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
  serialized_updated_ids: str
  duration: float
  speaker_id: int
  speaker_name: str
  lang: Language
  ds_name: str

  def get_speaker_name(self):
    return str(self.speaker_name)

class PreparedDataList(List[PreparedData]):
  def save(self, file_path: str):
    save_csv(self, file_path)

  def get_total_duration_s(self):
    x: PreparedData
    durations = [x.duration for x in self]
    total_duration = sum(durations)
    return total_duration

  def get_entry(self, i: int) -> PreparedData:
    for entry in self:
      if entry.i == i:
        return entry
    raise Exception(f"Entry {i} not found.")
  
  def get_random_entry(self) -> PreparedData:
    idx = random.choice(range(len(self)))
    return self[idx]

  def get_random_entry_ds_speaker(self, speaker_id: int) -> PreparedData:
    x: PreparedData
    relevant_entries = [x for x in self if x.speaker_id == speaker_id]
    assert len(relevant_entries)
    entry = random.choice(relevant_entries)
    return entry

  @staticmethod
  def get_key_for_sorting_after_entry_id(elem: PreparedData):
    return elem.entry_id

  @classmethod
  def load(cls, file_path: str):
    data = load_csv(file_path, PreparedData)
    return cls(data)


def preprocess(datasets: OrderedDict[str, Tuple[DsDataList, TextDataList, WavDataList, MelDataList, List[str], SymbolIdDict]], ds_speakers: List[Tuple[str, str]], speakers_as_accents: bool) -> Tuple[PreparedDataList, SymbolIdDict, SpeakersDict]:
  speakers_dict = {k: v[4] for k, v in datasets.items()}
  expanded_ds_speakers = expand_speakers(speakers_dict, ds_speakers)
  ds_speakers_list, speakers_id_dict = get_speakers(expanded_ds_speakers)
  ds_prepared_data: List[Tuple[PreparedData, SymbolIdDict]] = []

  for ds_name, dataset in datasets.items():
    ds_data, text_data, wav_data, mel_data, _, conv = dataset
    speaker_names = ds_speakers_list[ds_name]
    prep = get_prepared_data(ds_name, ds_data, speaker_names, text_data, wav_data, mel_data)
    ds_prepared_data.append((prep, conv))
  
  all_convs = [x[1] for x in ds_prepared_data]
  all_symbols = get_all_symbols(all_convs)
  final_conv = get_final_converter(all_symbols, list(speakers_id_dict.values()), merge=not speakers_as_accents)
  whole, conv = merge_prepared_data(ds_prepared_data, final_conv, use_subset_ids=speakers_as_accents)
  return whole, conv, speakers_id_dict

def map_to_prepared_data(ds_name: str, ds_data: DsData, text_data: TextData, wav_data: WavData, mel_data: MelData) -> PreparedData:
  prep_data = PreparedData(
    i=0,
    entry_id=ds_data.entry_id,
    basename=ds_data.basename,
    wav_path=wav_data.wav,
    mel_path=mel_data.mel_path,
    n_mel_channels=mel_data.n_mel_channels,
    serialized_updated_ids=text_data.serialized_symbol_ids,
    lang=text_data.lang,
    duration=wav_data.duration,
    speaker_id=ds_data.speaker_id,
    speaker_name=ds_data.get_speaker_name(),
    ds_name=ds_name
  )
  return prep_data

def get_prepared_data(ds_name: str, ds_data: DsDataList, speaker_names: List[Tuple[str, int]], text_list: TextDataList, wav_list: WavDataList, mel_list: MelDataList) -> PreparedDataList:
  res = PreparedDataList()
  counter = 0
  for speaker_name, new_speaker_id in speaker_names:
    ds_entry: DsData
    for ds_entry in ds_data:
      if ds_entry.get_speaker_name() == speaker_name:
        prep_data = map_to_prepared_data(
          ds_name=ds_name,
          ds_data=ds_entry,
          text_data=text_list[ds_entry.entry_id],
          wav_data=wav_list[ds_entry.entry_id],
          mel_data=mel_list[ds_entry.entry_id]
        )
        
        prep_data.i = counter
        prep_data.speaker_id = new_speaker_id

        res.append(prep_data)
        counter += 1

  res.sort(key=PreparedDataList.get_key_for_sorting_after_entry_id, reverse=False)
  return res

def get_all_symbols(converters: List[SymbolIdDict]) -> Set[str]:
  all_symbols = set()
  for conv in converters:
    all_symbols = all_symbols.union(set(conv.get_all_symbols()))
  return all_symbols

def get_final_converter(symbols: Set[str], speaker_ids: List[int], merge: bool) -> SymbolIdDict:
  # TODO: refactor
  # if merge:
  #   new_conv = SymbolIdDict.init_from_symbols(symbols)
  # else:
  #   new_conv = SymbolIdDict.init_from_symbols({})
  #   for speaker_id in speaker_ids:
  #     new_conv.add_symbols(symbols, ignore_existing=False, subset_id=speaker_id)
  # return new_conv
  pass

def merge_prepared_data(prep_list: List[Tuple[PreparedDataList, SymbolIdDict]], new_conv: SymbolIdDict, use_subset_ids: bool) -> Tuple[PreparedDataList, SymbolIdDict]:
  res = PreparedDataList()
  
  counter = 0
  prep_data_list: PreparedDataList
  for prep_data_list, conv in prep_list:
    entry: PreparedData
    for entry in prep_data_list:
      original_symbol_ids = SymbolIdDict.deserialize_symbol_ids(entry.serialized_updated_ids)
      original_symbols = conv.get_symbols(original_symbol_ids)
      subset_id = entry.speaker_id if use_subset_ids else 0
      # TODO: include accent
      #updated_symbol_ids = new_conv.symbols_to_ids(original_symbols, subset_id_if_multiple=subset_id, add_eos=False, replace_unknown_with_pad=True)
      updated_symbol_ids = new_conv.get_ids(original_symbols)
      serialized_updated_symbol_ids = SymbolIdDict.serialize_symbol_ids(updated_symbol_ids)
      entry.serialized_updated_ids = serialized_updated_symbol_ids
      entry.i = counter
      res.append(entry)
      counter += 1

  return res, new_conv
  
def split_prepared_data_train_test_val(prep: PreparedDataList, test_size: float, val_size: float, seed: int, shuffle: bool) -> Tuple[PreparedDataList, PreparedDataList, PreparedDataList]:
  speaker_data = {}
  data: PreparedData
  for data in prep:
    if data.speaker_id not in speaker_data:
      speaker_data[data.speaker_id] = []
    speaker_data[data.speaker_id].append(data)

  train, test, val = [], [], []
  for _, data in speaker_data.items():
    speaker_train, speaker_test, speaker_val = split_train_test_val(data, test_size, val_size, seed, shuffle=True)
    train.extend(speaker_train)
    test.extend(speaker_test)
    val.extend(speaker_val)

  return PreparedDataList(train), PreparedDataList(test), PreparedDataList(val)

def split_train_test_val(wholeset: list, test_size: float, validation_size: float, seed: int, shuffle: bool) -> (list, list, list):
  assert seed >= 0
  assert 0 <= test_size <= 1
  assert 0 <= validation_size <= 1
  assert test_size + validation_size < 1

  trainset, testset, valset = wholeset, [], []

  if validation_size:
    is_ok = assert_fraction_is_big_enough(validation_size, len(trainset))
    trainset, valset = train_test_split(trainset, test_size=validation_size, random_state=seed, shuffle=shuffle)
    if not is_ok:
      check_not_empty(trainset)
      check_not_empty(valset)
      print(f"Split was however successfull, trainsize {len(trainset)}, valsize: {len(valset)}.")
  if test_size:
    adj_test_size = test_size / (1 - validation_size)
    is_ok = assert_fraction_is_big_enough(adj_test_size, len(trainset))
    trainset, testset = train_test_split(trainset, test_size=adj_test_size, random_state=seed, shuffle=shuffle)
    if not is_ok:
      check_not_empty(trainset)
      check_not_empty(valset)
      print(f"Split was however successfull, trainsize {len(trainset)}, testsize: {len(testset)}.")

  return trainset, testset, valset

def check_not_empty(dataset: PreparedDataList):
  if not len(dataset):
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
  for ds, speaker_name in ds_speakers:
    if ds not in speakers_dict:
      continue
    if speaker_name == 'all':
      expanded_speakers.extend([(ds, speaker) for speaker in speakers_dict[ds]])
    else:
      if speaker_name not in speakers_dict[ds]:
        continue
      expanded_speakers.append((ds, speaker_name))
  expanded_speakers = list(sorted(set(expanded_speakers)))
  return expanded_speakers
