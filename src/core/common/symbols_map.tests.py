import os
import unittest
from src.core.common.symbols_map import SymbolsMap, get_symbols_id_mapping, create_symbols_map, create_inference_map_core, update_map
from src.core.common.symbol_id_dict import SymbolIdDict
from torch import nn
import tempfile

class UnitTests(unittest.TestCase):
  
  def test_from_two_sets(self):
    m = SymbolsMap.from_two_sets({"a", "b"}, {"b", "c"})

    self.assertEqual(2, len(m))
    self.assertEqual("b", m["b"])
    self.assertEqual("", m["c"])

  def test_update_map_empty_symbols_are_taken(self):
    old_map = SymbolsMap([
      ("a", "a"),
    ])

    new_map = SymbolsMap([
      ("a", ""),
      ("b", ""),
    ])

    res = update_map(old_map, new_map)

    self.assertEqual(2, len(new_map))
    self.assertEqual("a", new_map["a"])
    self.assertEqual("", new_map["b"])
    self.assertTrue(res)

  def test_update_map_non_existing_symbols_are_ignored(self):
    old_map = SymbolsMap([
      ("a", "a"),
    ])

    new_map = SymbolsMap([
      ("b", "b"),
    ])

    res = update_map(old_map, new_map)
    
    self.assertEqual(1, len(new_map))
    self.assertEqual("b", new_map["b"])
    self.assertTrue(res)

  def test_update_map_new_symbols_are_taken(self):
    old_map = SymbolsMap([
      ("a", "a"),
    ])

    new_map = SymbolsMap([
      ("a", "b"),
    ])

    res = update_map(old_map, new_map)

    self.assertEqual(1, len(new_map))
    self.assertEqual("b", new_map["a"])
    self.assertFalse(res)

  def test_update_map_new_symbols_are_added(self):
    old_map = SymbolsMap([
      ("a", "a"),
    ])

    new_map = SymbolsMap([
      ("a", "a"),
      ("b", "b"),
    ])

    res = update_map(old_map, new_map)

    self.assertEqual(2, len(new_map))
    self.assertEqual("a", new_map["a"])
    self.assertEqual("b", new_map["b"])
    self.assertTrue(res)

  def test_update_map(self):
    old_map = SymbolsMap([
      ("a", "a"),
      ("b", "c"),
      ("d", ""),
      ("g", "h"),
    ])

    new_map = SymbolsMap([
      ("a", "c"),
      ("b", "a"),
      ("d", "x"),
      ("e", "f"),
      ("g", ""),
    ])

    res = update_map(old_map, new_map)

    self.assertEqual("c", new_map["a"])
    self.assertEqual("a", new_map["b"])
    self.assertEqual("x", new_map["d"])
    self.assertEqual("f", new_map["e"])
    self.assertEqual("h", new_map["g"])
    self.assertTrue(res)

  def test_save_load_symbols_map(self):
    path = tempfile.mktemp()
    symbols_map = SymbolsMap([
      ("b", "a"),
      ("c", "b"),
      ("x", "y"),
    ])
    symbols_map.save(path)
    res = SymbolsMap.load(path)
    os.remove(path)
    self.assertEqual(3, len(res))
    self.assertEqual("a", res["b"])
    self.assertEqual("b", res["c"])
    self.assertEqual("y", res["x"])

  def test_create_inference_map_no_ipa(self):
    orig_symbols = {"b", "c"}
    corpora = "abc d\n \te"

    symbols_id_map, symbols = create_inference_map_core(orig_symbols, corpora, is_ipa=False)

    self.assertEqual(8, len(symbols_id_map))
    self.assertEqual("b", symbols_id_map["b"])
    self.assertEqual("c", symbols_id_map["c"])
    self.assertEqual("", symbols_id_map["\t"])
    self.assertEqual("", symbols_id_map["\n"])
    self.assertEqual("", symbols_id_map[" "])
    self.assertEqual("", symbols_id_map["a"])
    self.assertEqual("", symbols_id_map["d"])
    self.assertEqual("", symbols_id_map["e"])
    self.assertEqual(["b", "c"], symbols)

  def test_create_inference_map_ipa(self):
    orig_symbols = {"ŋ", "ɔ", " "}
    corpora = "ɛəŋ m\n \tɔ"

    symbols_id_map, symbols = create_inference_map_core(orig_symbols, corpora, is_ipa=True, ignore_arcs=True, ignore_tones=True)

    self.assertEqual(8, len(symbols_id_map))
    self.assertEqual(symbols_id_map["ŋ"], "ŋ")
    self.assertEqual(symbols_id_map["ɔ"], "ɔ")
    self.assertEqual(symbols_id_map[" "], " ")
    self.assertEqual(symbols_id_map["\t"], "")
    self.assertEqual(symbols_id_map["\n"], "")
    self.assertEqual(symbols_id_map["ə"], "")
    self.assertEqual(symbols_id_map["ɛ"], "")
    self.assertEqual(symbols_id_map["m"], "")
    self.assertEqual([" ", "ŋ", "ɔ"], symbols)

  def test_create_symbols_map_without_map(self):
    dest_symbols = {"b", "c"}
    orig_symbols = {"a", "b"}

    symbols_id_map = create_symbols_map(dest_symbols, orig_symbols)

    self.assertEqual(1, len(symbols_id_map))
    self.assertEqual(symbols_id_map["b"], "b")

  def test_create_symbols_map_with_map(self):
    dest_symbols = {"b", "c", "d"}
    orig_symbols = {"a", "b"}
    
    symbols_map = SymbolsMap.from_tuples([
      ("b", "a"),
      ("c", "b"),
      ("x", "y"),
    ])

    symbols_id_map = create_symbols_map(dest_symbols, orig_symbols, symbols_map)

    self.assertEqual(2, len(symbols_id_map))
    self.assertEqual(symbols_id_map["b"], "a")
    self.assertEqual(symbols_id_map["c"], "b")

  def test_get_symbols_id_mapping_without_map(self):
    model_conv = SymbolIdDict.init_from_symbols({"b", "c"})

    trained_symbols = SymbolIdDict.init_from_symbols({"a", "b"})

    symbols_id_map = get_symbols_id_mapping(
      dest_symbols=model_conv,
      orig_symbols=trained_symbols
    )

    self.assertEqual(3, len(symbols_id_map))
    self.assertEqual(symbols_id_map[model_conv.get_id("_")], trained_symbols.get_id("_"))
    self.assertEqual(symbols_id_map[model_conv.get_id("~")], trained_symbols.get_id("~"))
    self.assertEqual(symbols_id_map[model_conv.get_id("b")], trained_symbols.get_id("b"))

  def test_get_symbols_id_mapping_with_map(self):
    model_conv = SymbolIdDict.init_from_symbols({"b", "c", "d"})

    trained_symbols = SymbolIdDict.init_from_symbols({"a", "b"})
    symbols_map = SymbolsMap.from_tuples([
      ("b", "a"),
      ("c", "b"),
      ("x", "y"),
    ])

    symbols_id_map = get_symbols_id_mapping(model_conv, trained_symbols, symbols_map)

    self.assertEqual(2, len(symbols_id_map))
    self.assertEqual(symbols_id_map[model_conv.get_id("b")], trained_symbols.get_id("a"))
    self.assertEqual(symbols_id_map[model_conv.get_id("c")], trained_symbols.get_id("b"))

if __name__ == '__main__':
  suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
  unittest.TextTestRunner(verbosity=2).run(suite)
