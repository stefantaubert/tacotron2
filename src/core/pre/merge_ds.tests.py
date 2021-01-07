import unittest
from typing import List, OrderedDict, Tuple

from text_utils import AccentsDict
from text_utils import Gender
from text_utils import Language
from text_utils import SpeakersDict
from text_utils import SymbolIdDict
from src.core.pre.ds import DsData, DsDataList
from src.core.pre.mel import MelData, MelDataList
from src.core.pre.merge_ds import (DsDataset, PreparedData, PreparedDataList,
                                   expand_speakers, preprocess,
                                   split_prepared_data_train_test_val)
from src.core.pre.text.pre import TextData, TextDataList
from src.core.pre.wav import WavData, WavDataList


class UnitTests(unittest.TestCase):

  def test_ds_dataset_to_merged_dataset(self):
    ds = DsDataset(
      name="ds",
      accent_ids=AccentsDict.init_from_accents({"c", "d"}),
      data=DsDataList([DsData(
        entry_id=1,
        basename="basename",
        speaker_id=2,
        serialized_symbols="0,1",
        gender=Gender.MALE,
        lang=Language.GER,
        serialized_accents="0,0",
        wav_path="wav",
        speaker_name="speaker",
        text="text",
      )]),
      mels=MelDataList([MelData(
        entry_id=1,
        mel_path="mel",
        n_mel_channels=5,
      )]),
      speakers=SpeakersDict.fromlist(["sp1", "sp2", "sp3"]),
      symbol_ids=SymbolIdDict.init_from_symbols({"a", "b"}),
      texts=TextDataList([TextData(
        entry_id=1,
        lang=Language.IPA,
        serialized_accent_ids="1,0",
        serialized_symbol_ids="1,1",
        text="text_new",
      )]),
      wavs=WavDataList([WavData(
        entry_id=1,
        duration=15,
        sr=22050,
        wav="wav_new",
      )]),
    )

    res = ds_dataset_to_merged_dataset(ds)

    self.assertEqual("ds", res.name)
    self.assertEqual(2, len(res.accent_ids))
    self.assertEqual(3, len(res.speakers))
    self.assertEqual(2, len(res.symbol_ids))
    self.assertEqual(1, len(res.items()))
    first_entry = res.items()[0]
    self.assertEqual(1, first_entry.entry_id)
    self.assertEqual(Gender.MALE, first_entry.gender)
    self.assertEqual("basename", first_entry.basename)
    self.assertEqual(2, first_entry.speaker_id)
    self.assertEqual(Language.IPA, first_entry.lang)
    self.assertEqual("1,0", first_entry.serialized_accents)
    self.assertEqual("1,1", first_entry.serialized_symbols)
    self.assertEqual("wav_new", first_entry.wav_path)
    self.assertEqual(15, first_entry.duration)
    self.assertEqual(22050, first_entry.sampling_rate)
    self.assertEqual("mel", first_entry.mel_path)
    self.assertEqual(5, first_entry.n_mel_channels)

  def test_sort_prep_data_list(self):
    l = PreparedDataList([
      self.get_dummy_prep_data(entry_id=2),
      self.get_dummy_prep_data(entry_id=1),
      self.get_dummy_prep_data(entry_id=3)
    ])

    l.sort(key=PreparedDataList._get_key_for_sorting, reverse=False)

    self.assertEqual(1, l[0].entry_id)
    self.assertEqual(2, l[1].entry_id)
    self.assertEqual(3, l[2].entry_id)

  def test_expand_speakers(self):
    speakers = {
      "thchs": ["a1", "a2"],
      "ljs": ["1", "2"],
    }

    x = [("thchs", "all"), ("ljs", "2"), ("ljs", "2"), ("unknown", "??"), ("thchs", "??")]
    res = expand_speakers(speakers, x)
    self.assertEqual(3, len(res))
    self.assertEqual(("ljs", "2"), res[0])
    self.assertEqual(("thchs", "a1"), res[1])
    self.assertEqual(("thchs", "a2"), res[2])

  def test_get_speakers(self):
    x = [("thchs", "ab"), ("ljs", "23"), ("ljs", "12")]

    res, speakers_id_dict = get_speakers(x)

    self.assertEqual(2, len(res))
    self.assertTrue("ljs" in res.keys())
    self.assertTrue("thchs" in res.keys())
    self.assertEqual(("12", 0), res["ljs"][0])
    self.assertEqual(("23", 1), res["ljs"][1])
    self.assertEqual(("ab", 2), res["thchs"][0])
    self.assertEqual(["ljs,12", "ljs,23", "thchs,ab"], speakers_id_dict.get_all_speakers())

  def test_map_to_prepared_data(self):
    ds = DsData(1, "basename0", 11233, 0, "text0", "wavpath0", Language.ENG)
    text = TextData(1, "text_pre0", "12,45,65", Language.CHN)
    wav = WavData(1, "wavpath_pre0", 7.89, 22050)
    mel = MelData(1, "melpath_pre0", 80)

    res = map_to_prepared_data("ds1", ds, text, wav, mel)

    self.assertEqual(0, res.i)
    self.assertEqual(1, res.entry_id)
    self.assertEqual("basename0", res.basename)
    self.assertEqual("wavpath_pre0", res.wav_path)
    self.assertEqual("melpath_pre0", res.mel_path)
    self.assertEqual("12,45,65", res.serialized_symbol_ids)
    self.assertEqual(7.89, res.duration)
    self.assertEqual(0, res.speaker_id)
    self.assertEqual("11233", res.speaker_name)
    self.assertEqual(Language.CHN, res.lang)
    self.assertEqual("ds1", res.ds_name)

  def test_preprocess(self):
    datasets = {
      "ljs": (
        DsDataList([
          DsData(0, "basename0", "speaker0", 0, "text0", "wavpath0", Language.ENG),
          DsData(1, "basename0", "speaker0", 0, "text0", "wavpath0", Language.ENG),
          DsData(2, "basename0", "speaker1", 1, "text0", "wavpath0", Language.ENG),
        ]),
        TextDataList([
          TextData(0, "text_pre0", "1,2,3", Language.CHN),
          TextData(1, "text_pre0", "1,2,3", Language.CHN),
          TextData(2, "text_pre0", "1,2,3", Language.CHN),
        ]),
        WavDataList([
          WavData(0, "wavpath_pre0", 7.89, 22050),
          WavData(1, "wavpath_pre0", 7.89, 22050),
          WavData(2, "wavpath_pre0", 7.89, 22050),
        ]),
        MelDataList([
          MelData(0, "melpath_pre0", 80),
          MelData(1, "melpath_pre0", 80),
          MelData(2, "melpath_pre0", 80),
        ]),
        ["speaker0", "speaker1"],
        SymbolIdDict.init_from_symbols({"a", "b"})
      ),
      "thchs": (
        DsDataList([
          DsData(0, "basename0", "speaker0", 0, "text0", "wavpath0", Language.ENG),
          DsData(1, "basename0", "speaker1", 1, "text0", "wavpath0", Language.ENG),
          DsData(2, "basename0", "speaker1", 1, "text0", "wavpath0", Language.ENG),
        ]),
        TextDataList([
          TextData(0, "text_pre0", "1,2,3", Language.CHN),
          TextData(1, "text_pre0", "1,2,3", Language.CHN),
          TextData(2, "text_pre0", "1,2,3", Language.CHN),
        ]),
        WavDataList([
          WavData(0, "wavpath_pre0", 7.89, 22050),
          WavData(1, "wavpath_pre0", 7.89, 22050),
          WavData(2, "wavpath_pre0", 7.89, 22050),
        ]),
        MelDataList([
          MelData(0, "melpath_pre0", 80),
          MelData(1, "melpath_pre0", 80),
          MelData(2, "melpath_pre0", 80),
        ]),
        ["speaker0", "speaker1"],
        SymbolIdDict.init_from_symbols({"b", "c"})
      )
    }
    ds_speakers = {
      ("ljs", "speaker0"),
      ("thchs", "speaker1"),
    }

    whole, conv, speakers_id_dict = preprocess(datasets, ds_speakers, speakers_as_accents=False)

    self.assertEqual(4, len(whole))
    self.assertEqual(set({"a", "b", "c"}), set(conv.get_all_symbols()))
    # TODO
    self.assertEqual("1,2,3", whole[0].serialized_symbol_ids)
    self.assertEqual("1,2,3", whole[1].serialized_symbol_ids)
    self.assertEqual("1,3,4", whole[2].serialized_symbol_ids)
    self.assertEqual("1,3,4", whole[3].serialized_symbol_ids)
    self.assertEqual(["ljs,speaker0", "thchs,speaker1"], speakers_id_dict.get_speakers())

  def test_get_prepared_data_speaker_name_is_int(self):
    speaker_names: List[Tuple[str, int]] = [("1123", 15)]
    ds_data = DsDataList([
      DsData(0, "", 1123, 0, "", "", 0),
    ])
    text_list = TextDataList([
      TextData(0, "", "", 0),
    ])
    wav_list = WavDataList([
      WavData(0, "", 0, 0),
    ])
    mel_list = MelDataList([
      MelData(0, "", 0),
    ])

    res: PreparedDataList = get_prepared_data(
      "", ds_data, speaker_names, text_list, wav_list, mel_list)

    self.assertEqual(1, len(res))
    self.assertEqual("1123", res[0].speaker_name)

  def test_get_prepared_data(self):
    speaker_names: List[Tuple[str, int]] = [("speaker1", 15)]
    ds_data = DsDataList([
      DsData(0, "basename0", "speaker0", 0, "text0", "wavpath0", Language.ENG),
      DsData(1, "basename1", "speaker1", 1, "text1", "wavpath1", Language.ENG),
      DsData(2, "basename2", "speaker1", 2, "text2", "wavpath2", Language.ENG),
    ])
    text_list = TextDataList([
      TextData(0, "text_pre0", "12,45,65", Language.CHN),
      TextData(1, "text_pre0", "65,78,16", Language.CHN),
      TextData(2, "text_pre0", "66,78,16", Language.CHN),
    ])
    wav_list = WavDataList([
      WavData(0, "wavpath_pre0", 7.89, 22050),
      WavData(1, "wavpath_pre1", 8.98, 22050),
      WavData(2, "wavpath_pre2", 9.98, 22050),
    ])
    mel_list = MelDataList([
      MelData(0, "melpath_pre0", 80),
      MelData(1, "melpath_pre1", 80),
      MelData(2, "melpath_pre2", 80),
    ])

    res: PreparedDataList = get_prepared_data(
      "ds1", ds_data, speaker_names, text_list, wav_list, mel_list)

    self.assertEqual(2, len(res))

    self.assertEqual(0, res[0].i)
    self.assertEqual(1, res[0].entry_id)
    self.assertEqual(15, res[0].speaker_id)
    self.assertEqual("basename1", res[0].basename)
    self.assertEqual("65,78,16", res[0].serialized_symbol_ids)
    self.assertEqual("wavpath_pre1", res[0].wav_path)
    self.assertEqual("melpath_pre1", res[0].mel_path)
    self.assertEqual("ds1", res[0].ds_name)

    self.assertEqual(1, res[1].i)
    self.assertEqual(2, res[1].entry_id)
    self.assertEqual(15, res[1].speaker_id)
    self.assertEqual("basename2", res[1].basename)
    self.assertEqual("66,78,16", res[1].serialized_symbol_ids)
    self.assertEqual("wavpath_pre2", res[1].wav_path)
    self.assertEqual("melpath_pre2", res[1].mel_path)
    self.assertEqual("ds1", res[1].ds_name)

  def test_merge_prepared_data(self):
    prep_list = [
      (PreparedDataList([
        PreparedData(0, 1, "", "", "", 0, "0,1,2", 0, 0, "", 0, ""),
      ]),
          SymbolIdDict({
              0: (0, "a"),
              1: (0, "b"),
              2: (0, "c"),
          })),
      (PreparedDataList([
        PreparedData(0, 2, "", "", "", 0, "0,1,2", 0, 0, "", 0, ""),
      ]),
          SymbolIdDict({
              0: (0, "b"),
              1: (0, "a"),
              2: (0, "d"),
          })),
    ]

    res, conv = merge_prepared_data(prep_list)

    self.assertEqual(6, len(conv))
    self.assertEqual("todo", conv.get_symbol(0))
    self.assertEqual("todo", conv.get_symbol(1))
    self.assertEqual("a", conv.get_symbol(2))
    self.assertEqual("b", conv.get_symbol(3))
    self.assertEqual("c", conv.get_symbol(4))
    self.assertEqual("d", conv.get_symbol(5))

    self.assertEqual(2, len(res))
    self.assertEqual(0, res[0].i)
    self.assertEqual(1, res[1].i)
    self.assertEqual(1, res[0].entry_id)
    self.assertEqual(2, res[1].entry_id)
    self.assertEqual("2,3,4", res[0].serialized_symbol_ids)
    self.assertEqual("3,2,5", res[1].serialized_symbol_ids)

  # def test_split_train_test_val_low_data_count_ignores_testset_and_valset(self):
  #   data = [1]

  #   train, test, val = split_train_test_val(data, test_size=1/3, validation_size=1/3, seed=0, shuffle=False)

  #   self.assertEqual([1], train)
  #   self.assertEqual([], test)
  #   self.assertEqual([], val)

  # def test_split_train_test_val_3_elements(self):
  #   data = [1, 2, 3]

  #   train, test, val = split_train_test_val(data, test_size=1/3, validation_size=1/3, seed=0, shuffle=False)

  #   self.assertEqual([1], train)
  #   self.assertEqual([2], test)
  #   self.assertEqual([3], val)

  # def test_split_train_test_val_low_data_count_ignores_testset(self):
  #   data = [1, 2]

  #   train, test, val = split_train_test_val(data, test_size=1/3, validation_size=1/3, seed=0, shuffle=False)

  #   self.assertEqual([1], train)
  #   self.assertEqual([], test)
  #   self.assertEqual([2], val)

  def test_split_prepared_data(self):
    data = PreparedDataList([
      self.get_dummy_prep_data(i=0),
      self.get_dummy_prep_data(i=1),
      self.get_dummy_prep_data(i=2),
      self.get_dummy_prep_data(i=3),
      self.get_dummy_prep_data(i=4),
      self.get_dummy_prep_data(i=5),
      self.get_dummy_prep_data(i=6),
      self.get_dummy_prep_data(i=7),
      self.get_dummy_prep_data(i=8),
      self.get_dummy_prep_data(i=9),
      self.get_dummy_prep_data(i=10),
      self.get_dummy_prep_data(i=11),
    ])

    train, test, val = split_prepared_data_train_test_val(
      data, test_size=1 / 6, validation_size=2 / 6, seed=0, shuffle=False)

    self.assertEqual(2, len(test))
    self.assertEqual(4, len(val))
    self.assertEqual(6, len(train))

  @staticmethod
  def get_dummy_prep_data(i: int = 0, entry_id: int = 0):
    return PreparedData(i, entry_id, "", "", "", 0, "", 0, 1, "", Language.ENG, "")


if __name__ == '__main__':
  suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
  unittest.TextTestRunner(verbosity=2).run(suite)
