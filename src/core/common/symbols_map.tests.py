import os
import tempfile
import unittest
from logging import getLogger

from src.core.common.symbol_id_dict import SymbolIdDict
from src.core.common.symbols_map import SymbolsMap, create_or_update_map

test_logger = getLogger("Tests")


class UnitTests(unittest.TestCase):

  def test_from_two_sets(self):
    m = SymbolsMap.from_intersection({"a", "b"}, {"b", "c"})

    self.assertEqual(2, len(m))
    self.assertEqual("b", m["b"])
    self.assertEqual("", m["c"])

  def test_update_existing_to_mappings_overwrites_empty_mapping(self):
    old = SymbolsMap({"a": ""})

    old.update_existing_to_mappings({"a": "a"})

    self.assertEqual(1, len(old))
    self.assertEqual("a", old["a"])

  def test_update_existing_to_mappings_ignores_mapping_with_unknown_symbol(self):
    old = SymbolsMap({"b": ""})

    old.update_existing_to_mappings({"a": "a"})

    self.assertEqual(1, len(old))
    self.assertEqual("", old["b"])

  def test_update_existing_to_mappings_ignores_mapping_with_empty_symbol(self):
    old = SymbolsMap({"a": "a"})

    old.update_existing_to_mappings({"a": ""})

    self.assertEqual(1, len(old))
    self.assertEqual("a", old["a"])

  # def test_update_map_non_existing_symbols_are_ignored(self):
  #   old_map = SymbolsMap([
  #     ("a", "a"),
  #   ])

  #   new_map = SymbolsMap([
  #     ("b", "b"),
  #   ])

  #   res = update_map(old_map, new_map)

  #   self.assertEqual(1, len(new_map))
  #   self.assertEqual("b", new_map["b"])
  #   self.assertTrue(res)

  # def test_update_map_new_symbols_are_taken(self):
  #   old_map = SymbolsMap([
  #     ("a", "a"),
  #   ])

  #   new_map = SymbolsMap([
  #     ("a", "b"),
  #   ])

  #   res = update_map(old_map, new_map)

  #   self.assertEqual(1, len(new_map))
  #   self.assertEqual("b", new_map["a"])
  #   self.assertFalse(res)

  # def test_update_map_new_symbols_are_added(self):
  #   old_map = SymbolsMap([
  #     ("a", "a"),
  #   ])

  #   new_map = SymbolsMap([
  #     ("a", "a"),
  #     ("b", "b"),
  #   ])

  #   res = update_map(old_map, new_map)

  #   self.assertEqual(2, len(new_map))
  #   self.assertEqual("a", new_map["a"])
  #   self.assertEqual("b", new_map["b"])
  #   self.assertTrue(res)

  # def test_update_map(self):
  #   old_map = SymbolsMap([
  #     ("a", "a"),
  #     ("b", "c"),
  #     ("d", ""),
  #     ("g", "h"),
  #   ])

  #   new_map = SymbolsMap([
  #     ("a", "c"),
  #     ("b", "a"),
  #     ("d", "x"),
  #     ("e", "f"),
  #     ("g", ""),
  #   ])

  #   res = update_map(old_map, new_map)

  #   self.assertEqual("c", new_map["a"])
  #   self.assertEqual("a", new_map["b"])
  #   self.assertEqual("x", new_map["d"])
  #   self.assertEqual("f", new_map["e"])
  #   self.assertEqual("h", new_map["g"])
  #   self.assertTrue(res)

  def test_save_load_symbols_map(self):
    path = tempfile.mktemp()
    symbols_map = SymbolsMap({
      "b": "a",
      "c": "b",
      "x": "y",
    })
    symbols_map.save(path)
    res = SymbolsMap.load(path)
    os.remove(path)
    self.assertEqual(3, len(res))
    self.assertEqual("a", res["b"])
    self.assertEqual("b", res["c"])
    self.assertEqual("y", res["x"])

  def test_create_or_update_map_no_other_maps(self):
    orig_symbols = {"b", "c"}
    corpora = {"a", "b"}

    symbols_id_map, symbols = create_or_update_map(
      orig=orig_symbols,
      dest=corpora,
      existing_map=None,
      template_map=None,
    )

    self.assertEqual(symbols_id_map["a"], "")
    self.assertEqual(symbols_id_map["b"], "b")
    self.assertEqual(2, len(symbols_id_map))
    self.assertEqual(["b", "c"], symbols)

  def test_create_or_update_map_with_template_map(self):
    orig_symbols = {"c", "d", "e"}
    corpora = {"a", "b", "c"}

    template_map = SymbolsMap({
      "b": "d",
      "e": "f"
    })

    symbols_id_map, symbols = create_or_update_map(
      orig=orig_symbols,
      dest=corpora,
      template_map=template_map,
      existing_map=None,
    )

    self.assertEqual(symbols_id_map["a"], "")
    self.assertEqual(symbols_id_map["b"], "d")
    self.assertEqual(symbols_id_map["c"], "c")
    self.assertEqual(3, len(symbols_id_map))
    self.assertEqual(["c", "d", "e"], symbols)

  def test_create_or_update_map_with_template_map_and_existing_map(self):
    orig_symbols = {"c", "d", "e"}
    corpora = {"a", "b", "c"}

    existing_map = SymbolsMap({
      "a": "a",
      "b": "b",
      "c": "c",
    })

    template_map = SymbolsMap({
      "b": "d",
      "e": "f"
    })

    symbols_id_map, symbols = create_or_update_map(
      orig=orig_symbols,
      dest=corpora,
      template_map=template_map,
      existing_map=None,
    )

    self.assertEqual(symbols_id_map["a"], "")
    self.assertEqual(symbols_id_map["b"], "d")
    self.assertEqual(symbols_id_map["c"], "c")
    self.assertEqual(3, len(symbols_id_map))
    self.assertEqual(["c", "d", "e"], symbols)

  def test_get_symbols_id_mapping_without_map(self):
    from_symbols = {"b", "c"}
    to_symbols = {"a", "b"}
    from_conv = SymbolIdDict.init_from_symbols(from_symbols)
    to_conv = SymbolIdDict.init_from_symbols(to_symbols)
    mapping = SymbolsMap.from_intersection(from_symbols, to_symbols)
    mapping.update_existing_to_mappings({"a": "c"})

    symbols_id_map = mapping.convert_to_symbols_ids_map(
      to_symbols=to_conv,
      from_symbols=from_conv,
    )

    self.assertEqual(symbols_id_map[from_conv.get_id("b")], to_conv.get_id("b"))
    self.assertEqual(symbols_id_map[from_conv.get_id("c")], to_conv.get_id("a"))
    self.assertEqual(2, len(symbols_id_map))


if __name__ == '__main__':
  suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
  unittest.TextTestRunner(verbosity=2).run(suite)
