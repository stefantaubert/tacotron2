import unittest
from logging import getLogger

from src.core.common import (AccentsDict, Language, SpeakersDict,
                             SpeakersLogDict, SymbolIdDict,
                             remove_duplicates_list_orderpreserving)
from src.core.pre.ds import DsData
from src.core.pre.text.pre import *


class UnitTests(unittest.TestCase):
  def test_preprocess(self):
    data = DsDataList([
      DsData(5, "bn", "spk", 1, "text", "1,0", "0,1", "wav", Language.ENG)
    ])
    symbol_ids = SymbolIdDict.init_from_symbols({"a", "b"})

    result, conv, symbols_dict = preprocess(data, symbol_ids)

    self.assertEqual(2, len(conv))
    self.assertEqual(1, len(result))
    self.assertEqual(2, len(symbols_dict))

    self.assertEqual(5, result[0].entry_id)
    self.assertEqual(Language.ENG, result[0].lang)
    self.assertEqual("1,0", result[0].serialized_symbol_ids)
    self.assertEqual("0,1", result[0].serialized_accent_ids)
    self.assertEqual("ba", result[0].text)

  def test_normalize(self):
    data = TextDataList([
      TextData(5, "  b ", "0,0,1,0", "7,7,7,7", Language.ENG)
    ])
    symbol_ids = SymbolIdDict.init_from_symbols({" ", "b"})

    result, conv, symbols_dict = normalize(data, symbol_ids, getLogger())

    self.assertEqual(1, len(conv))
    self.assertEqual(1, len(result))
    self.assertEqual(1, len(symbols_dict))

    self.assertEqual(5, result[0].entry_id)
    self.assertEqual(Language.ENG, result[0].lang)
    self.assertEqual("0", result[0].serialized_symbol_ids)
    self.assertEqual("7", result[0].serialized_accent_ids)
    self.assertEqual("b", result[0].text)

  def test_ipa(self):
    data = TextDataList([
      TextData(5, "b", "1", "7", Language.ENG)
    ])
    symbol_ids = SymbolIdDict.init_from_symbols({" ", "b", "c"})

    result, conv, symbols_dict = convert_to_ipa(
      data, symbol_ids, ignore_tones=False, ignore_arcs=False)

    self.assertEqual(2, len(conv))
    self.assertEqual(1, len(result))
    self.assertEqual(2, len(symbols_dict))

    self.assertEqual(5, result[0].entry_id)
    self.assertEqual(Language.IPA, result[0].lang)
    self.assertEqual("0,1", result[0].serialized_symbol_ids)
    self.assertEqual("7,7", result[0].serialized_accent_ids)
    self.assertEqual("bi", result[0].text)


if __name__ == '__main__':
  suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
  unittest.TextTestRunner(verbosity=2).run(suite)
