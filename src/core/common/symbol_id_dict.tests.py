import unittest

from src.core.common.symbol_id_dict import *


class UnitTests(unittest.TestCase):

  def test_remove_symbol_ids(self):
    res = SymbolIdDict.init_from_symbols({"a", "b", "c"})

    res.remove_ids({0, 2})
    self.assertEqual(1, len(res))
    self.assertEqual("b", res.get_symbol(1))
    self.assertEqual(1, res.get_id("b"))

  def test_init_from_symbols_adds_no_symbols(self):
    res = SymbolIdDict.init_from_symbols({"a", "b", "c"})

    self.assertEqual(3, len(res))

  def test_init_from_symbols_is_sorted(self):
    res = SymbolIdDict.init_from_symbols({"c", "a", "b"})

    self.assertEqual("a", res.get_symbol(0))
    self.assertEqual("b", res.get_symbol(1))
    self.assertEqual("c", res.get_symbol(2))

  def test_get_text_from_serialized_ids(self):
    symbol_ids = SymbolIdDict.init_from_symbols({"a", "b"})

    res = symbol_ids.get_text("0,1,1")

    self.assertEqual("abb", res)

  def test_get_text_from_ids(self):
    symbol_ids = SymbolIdDict.init_from_symbols({"a", "b"})

    res = symbol_ids.get_text([0, 1, 1])

    self.assertEqual("abb", res)

  def test_init_from_symbols_with_pad_uses_pad_const(self):
    res = SymbolIdDict.init_from_symbols_with_pad({"b", "a"})

    self.assertEqual(PADDING_SYMBOL, res.get_symbol(0))
    self.assertEqual("a", res.get_symbol(1))
    self.assertEqual("b", res.get_symbol(2))

  def test_init_from_symbols_with_pad_has_pad_at_idx_zero(self):
    res = SymbolIdDict.init_from_symbols_with_pad({"b", "a"}, "xx")

    self.assertEqual("xx", res.get_symbol(0))
    self.assertEqual("a", res.get_symbol(1))
    self.assertEqual("b", res.get_symbol(2))

  def test_init_from_symbols_with_pad_ignores_existing_pad(self):
    res = SymbolIdDict.init_from_symbols_with_pad({"b", "a", "xx"}, "xx")

    self.assertEqual("xx", res.get_symbol(0))
    self.assertEqual("a", res.get_symbol(1))
    self.assertEqual("b", res.get_symbol(2))


if __name__ == '__main__':
  suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
  unittest.TextTestRunner(verbosity=2).run(suite)
