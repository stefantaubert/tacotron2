import unittest
from src.core.pre.ds import SpeakersDict, DsDataList, DsData
from src.core.pre.text import TextDataList, TextData
from src.core.pre.wav import WavDataList, WavData
from src.core.pre.mel import MelDataList, MelData
from src.core.pre.merge_ds import expand_speakers, get_speakers, get_prepared_data, PreparedData, PreparedDataList, map_to_prepared_data, merge_prepared_data, split_prepared_data_train_test_val, split_train_test_val
from src.core.pre.ds import SpeakersDict
from src.core.pre.language import Language
from typing import List, Tuple, OrderedDict
from src.text.symbol_converter import SymbolConverter


class UnitTests(unittest.TestCase):
  
  def test_expand_speakers(self):
    speakers = {
      "thchs": SpeakersDict({
        "a1": 0,
        "a2": 1,
      }),
      "ljs": SpeakersDict({
        "1": 0,
        "2": 1,
      })
    }
    x = [("thchs","all"), ("ljs", "2"), ("ljs", "2")]
    res = expand_speakers(speakers, x)
    self.assertEqual(3, len(res))
    self.assertEqual(("ljs", "2"), res[0])
    self.assertEqual(("thchs", "a1"), res[1])
    self.assertEqual(("thchs", "a2"), res[2])

  def test_get_speakers(self):
    x = [("thchs","a"), ("ljs", "1"), ("ljs", "2")]
    res = get_speakers(x)
    self.assertEqual(2, len(res))
    self.assertTrue("ljs" in res.keys())
    self.assertTrue("thchs" in res.keys())
    self.assertEqual(("a", 0), res["thchs"][0])
    self.assertEqual(("1", 1), res["ljs"][0])
    self.assertEqual(("2", 2), res["ljs"][1])

  def test_map_to_prepared_data(self):
    ds = DsData(1, "basename0", "speaker0", 0, "text0", "wavpath0", Language.ENG)
    text = TextData(1, "text_pre0", "12,45,65", Language.CHN)
    wav = WavData(1, "wavpath_pre0", 7.89, 22050)
    mel = MelData(1, "melpath_pre0", 7.89, 22050)

    res = map_to_prepared_data(ds, text, wav, mel)

    self.assertEqual(0, res.i)
    self.assertEqual(1, res.entry_id)
    self.assertEqual("basename0", res.basename)
    self.assertEqual("wavpath_pre0", res.wav_path)
    self.assertEqual("melpath_pre0", res.mel_path)
    self.assertEqual("12,45,65", res.serialized_updated_ids)
    self.assertEqual(7.89, res.duration)
    self.assertEqual(0, res.speaker_id)
    self.assertEqual("speaker0", res.speaker_name)
    self.assertEqual(Language.CHN, res.lang)

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
      MelData(0, "melpath_pre0", 7.89, 22050),
      MelData(1, "melpath_pre1", 8.98, 22050),
      MelData(2, "melpath_pre2", 9.98, 22050),
    ])

    res: PreparedDataList = get_prepared_data(ds_data, speaker_names, text_list, wav_list, mel_list)

    self.assertEqual(2, len(res))

    self.assertEqual(0, res[0].i)
    self.assertEqual(1, res[0].entry_id)
    self.assertEqual(15, res[0].speaker_id)
    self.assertEqual("basename1", res[0].basename)
    self.assertEqual("65,78,16", res[0].serialized_updated_ids)
    self.assertEqual("wavpath_pre1", res[0].wav_path)
    self.assertEqual("melpath_pre1", res[0].mel_path)

    self.assertEqual(1, res[1].i)
    self.assertEqual(2, res[1].entry_id)
    self.assertEqual(15, res[1].speaker_id)
    self.assertEqual("basename2", res[1].basename)
    self.assertEqual("66,78,16", res[1].serialized_updated_ids)
    self.assertEqual("wavpath_pre2", res[1].wav_path)
    self.assertEqual("melpath_pre2", res[1].mel_path)

  def test_merge_prepared_data(self):
    prep_list = [
      (PreparedDataList([
        PreparedData(0, 1, "basename1", "wav1", "mel1", "0,1,2", 4.5, 15, "speaker1", Language.ENG),
      ]),
      SymbolConverter({
        0: (0, "a"),
        1: (0, "b"),
        2: (0, "c"),
      })),
      (PreparedDataList([
        PreparedData(0, 2, "basename2", "wav2", "mel2", "0,1,2", 5.5, 16, "speaker2", Language.CHN),
      ]),
      SymbolConverter({
        0: (0, "b"),
        1: (0, "a"),
        2: (0, "d"),
      })),
    ]

    res, conv = merge_prepared_data(prep_list)

    self.assertEqual(6, conv.get_symbol_ids_count())
    self.assertEqual("_", conv.id_to_symbol(0))
    self.assertEqual("~", conv.id_to_symbol(1))
    self.assertEqual("a", conv.id_to_symbol(2))
    self.assertEqual("b", conv.id_to_symbol(3))
    self.assertEqual("c", conv.id_to_symbol(4))
    self.assertEqual("d", conv.id_to_symbol(5))

    self.assertEqual(2, len(res))
    self.assertEqual(1, res[0].entry_id)
    self.assertEqual(2, res[1].entry_id)
    self.assertEqual("2,3,4", res[0].serialized_updated_ids)
    self.assertEqual("3,2,5", res[1].serialized_updated_ids)

  def test_split_train_test_val_123(self):
    data = [0] * 6

    train, test, val = split_train_test_val(data, test_size=1/6, validation_size=2/6, seed=0)

    self.assertEqual(3, len(train))
    self.assertEqual(1, len(test))
    self.assertEqual(2, len(val))

  def test_split_train_test_val_024(self):
    data = [0] * 6

    train, test, val = split_train_test_val(data, test_size=0, validation_size=2/6, seed=0)

    self.assertEqual(4, len(train))
    self.assertEqual(0, len(test))
    self.assertEqual(2, len(val))

  def test_split_train_test_val_510(self):
    data = [0] * 6

    train, test, val = split_train_test_val(data, test_size=1/6, validation_size=0, seed=0)

    self.assertEqual(5, len(train))
    self.assertEqual(1, len(test))
    self.assertEqual(0, len(val))

  def test_split_prepared_data(self):
    data = PreparedDataList([
      PreparedData(0, 0, "", "", "", "", 0, 1, "", Language.ENG),
      PreparedData(1, 0, "", "", "", "", 0, 1, "", Language.ENG),
      PreparedData(2, 0, "", "", "", "", 0, 1, "", Language.ENG),
      PreparedData(3, 0, "", "", "", "", 0, 1, "", Language.ENG),
      PreparedData(4, 0, "", "", "", "", 0, 1, "", Language.ENG),
      PreparedData(5, 0, "", "", "", "", 0, 1, "", Language.ENG),
      PreparedData(6, 0, "", "", "", "", 0, 2, "", Language.ENG),
      PreparedData(7, 0, "", "", "", "", 0, 2, "", Language.ENG),
      PreparedData(8, 0, "", "", "", "", 0, 2, "", Language.ENG),
      PreparedData(9, 0, "", "", "", "", 0, 2, "", Language.ENG),
      PreparedData(10, 0, "", "", "", "", 0, 2, "", Language.ENG),
      PreparedData(11, 0, "", "", "", "", 0, 2, "", Language.ENG),
    ])

    train, test, val = split_prepared_data_train_test_val(data, test_size=1/6, val_size=2/6, seed=0)

    self.assertEqual(2, len(test))
    self.assertEqual(4, len(val))
    self.assertEqual(6, len(train))

if __name__ == '__main__':
  suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
  unittest.TextTestRunner(verbosity=2).run(suite)
