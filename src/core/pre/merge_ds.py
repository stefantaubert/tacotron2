
from src.core.pre.ds import SpeakersDict, DsDataList, DsData
from src.core.pre.text import TextDataList, TextData
from src.core.pre.wav import WavDataList, WavData
from src.core.pre.mel import MelDataList, MelData
from typing import List, Tuple, OrderedDict
from dataclasses import dataclass
from src.core.pre.language import Language
from src.text.symbol_converter import SymbolConverter
from sklearn.model_selection import train_test_split

@dataclass()
class PreparedData:
  i: int
  entry_id: int
  basename: str
  wav_path: str
  mel_path: str
  serialized_updated_ids: str
  duration: float
  speaker_id: int
  speaker_name: str
  lang: Language

class PreparedDataList(List[PreparedData]):
  pass

# def filter_speaker(l: DsDataList, speaker_name: str) -> DsDataList:
#   x: DsData
#   res = DsDataList()
#   for x in l:
#     if x.speaker_name == speaker_name:
#       res.append(x)
#   return res

def preprocess(datasets: OrderedDict[str, Tuple[DsDataList, TextDataList, WavDataList, MelDataList, SpeakersDict, SymbolConverter]], ds_speakers: OrderedDict[str, List[str]], test_size: float, val_size: float, seed: int) -> Tuple[PreparedDataList, PreparedDataList, PreparedDataList, PreparedDataList, SymbolConverter]:
  speakers_dicts = {k: v[4] for k, v in datasets}
  expanded_ds_speakers = expand_speakers(speakers_dicts, ds_speakers)
  ds_speakers_list = get_speakers(expanded_ds_speakers)
  ds_prepared_data: List[Tuple[PreparedData, SymbolConverter]]

  for ds_name, dataset in datasets.items():
    ds_data, text_data, wav_data, mel_data, _, conv = dataset
    speaker_names = ds_speakers_list[ds_name]
    prep = get_prepared_data(ds_data, speaker_names, text_data, wav_data, mel_data)
    ds_prepared_data.append((prep, conv))
  
  whole, conv = merge_prepared_data(ds_prepared_data)
  train, val, test = split_prepared_data_train_test_val(whole, test_size, val_size, seed)
  return whole, train, val, test, conv

def map_to_prepared_data(ds_data: DsData, text_data: TextData, wav_data: WavData, mel_data: MelData) -> PreparedData:
  prep_data = PreparedData(
    i=0,
    entry_id=ds_data.entry_id,
    basename=ds_data.basename,
    wav_path=wav_data.wav,
    mel_path=mel_data.mel_path,
    serialized_updated_ids=text_data.serialized_symbol_ids,
    lang=text_data.lang,
    duration=wav_data.duration,
    speaker_id=ds_data.speaker_id,
    speaker_name=ds_data.speaker_name
  )
  return prep_data

def get_prepared_data(ds_data: DsDataList, speaker_names: List[Tuple[str, int]], text_list: TextDataList, wav_list: WavDataList, mel_list: MelDataList) -> PreparedDataList:
  res = PreparedDataList()
  counter = 0
  for speaker_name, new_speaker_id in speaker_names:
    ds_entry: DsData
    for ds_entry in ds_data:
      if ds_entry.speaker_name == speaker_name:
        prep_data = map_to_prepared_data(
          ds_data=ds_entry,
          text_data=text_list[ds_entry.entry_id],
          wav_data=wav_list[ds_entry.entry_id],
          mel_data=mel_list[ds_entry.entry_id]
        )
        
        prep_data.i = counter
        prep_data.speaker_id = new_speaker_id

        res.append(prep_data)
        counter += 1
  return res

def merge_prepared_data(prep_list: List[Tuple[PreparedDataList, SymbolConverter]]) -> Tuple[PreparedDataList, SymbolConverter]:
  res = PreparedDataList()
  all_symbols = []
  prep_data_list: PreparedDataList
  
  all_symbols = set()
  for _, conv in prep_list:
    all_symbols = all_symbols.union(set(conv.get_symbols(include_id=False, include_subset_id=False)))
  
  new_conv = SymbolConverter.init_from_symbols(all_symbols)

  for prep_data_list, conv in prep_list:
    entry: PreparedData
    for entry in prep_data_list:
      original_symbol_ids = SymbolConverter.deserialize_symbol_ids(entry.serialized_updated_ids)
      original_symbols = conv.ids_to_symbols(original_symbol_ids)
      updated_symbol_ids = new_conv.symbols_to_ids(original_symbols, subset_id_if_multiple=0, add_eos=False, replace_unknown_with_pad=True)
      serialized_updated_symbol_ids = SymbolConverter.serialize_symbol_ids(updated_symbol_ids)
      entry.serialized_updated_ids = serialized_updated_symbol_ids
      res.append(entry)

  return res, new_conv
  
def split_prepared_data_train_test_val(prep: PreparedDataList, test_size: float, val_size: float, seed: int) -> Tuple[PreparedDataList, PreparedDataList, PreparedDataList]:
  speaker_data = {}
  data: PreparedData
  for data in prep:
    if data.speaker_id not in speaker_data:
      speaker_data[data.speaker_id] = []
    speaker_data[data.speaker_id].append(data)

  train, test, val = [], [], []
  for speaker, data in speaker_data.items():
    speaker_train, speaker_test, speaker_val = split_train_test_val(data, test_size, val_size, seed)
    train.extend(speaker_train)
    test.extend(speaker_test)
    val.extend(speaker_val)

  return train, test, val

def split_train_test_val(wholeset: list, test_size: float, validation_size: float, seed: int) -> (list, list, list):
  assert seed > 0
  assert 0 <= test_size <= 1
  assert 0 <= validation_size <= 1

  trainset, testset, valset = wholeset, [], []

  rest_size = validation_size + test_size
  trainset, rest = train_test_split(wholeset, train_size=(1 - rest_size), random_state=seed)

  if rest_size:
    validation_ratio = validation_size / rest_size
    if validation_ratio:
      if validation_ratio == 1:
        valset = rest
      else:
        valset, testset = train_test_split(rest, train_size=validation_ratio, random_state=seed)
    else:
      testset = rest

  return trainset, testset, valset

def get_speakers(ds_speakers: Tuple[str, List[str]]) -> OrderedDict[str, List[Tuple[str, int]]]:
  # ljs,1;thchs,A12;thchs,A13,thchs,all
  res = {
    "ljs": [("A12", 0), ("A13", 1)],
    "thchs": [("B12", 2)],
  }
  res = OrderedDict()
  counter = 0
  for ds_name, speaker_names in ds_speakers:
    for speaker_name in speaker_names:
      if ds_name not in res:
        res[ds_name] = []
      res[ds_name].append((speaker_name, counter))
      counter += 1

  return res

def expand_speakers(speakers_dict: OrderedDict[str, SpeakersDict], ds_speakers: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
  # expand all
  expanded_speakers: List[Tuple[str, str]] = []
  for ds, speaker_name in ds_speakers:
    if speaker_name == 'all':
      expanded_speakers.extend([(ds, k) for k, v in speakers_dict[ds].items()])
    else:
      expanded_speakers.append((ds, speaker_name))
  expanded_speakers = list(sorted(set(expanded_speakers)))
  return expanded_speakers
