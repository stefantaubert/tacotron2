import unittest

from src.core.common.accents_dict import AccentsDict
from src.core.common.globals import PADDING_ACCENT


class UnitTests(unittest.TestCase):

  def test_init_from_accents_adds_no_accents(self):
    res = AccentsDict.init_from_accents({"a", "b", "c"})

    self.assertEqual(3, len(res))

  def test_init_from_accents_is_sorted(self):
    res = AccentsDict.init_from_accents({"c", "a", "b"})

    self.assertEqual("a", res.get_accent(0))
    self.assertEqual("b", res.get_accent(1))
    self.assertEqual("c", res.get_accent(2))

  def test_init_from_accents_with_pad_uses_pad_const(self):
    res = AccentsDict.init_from_accents_with_pad({"b", "a"})

    self.assertEqual(PADDING_ACCENT, res.get_accent(0))
    self.assertEqual("a", res.get_accent(1))
    self.assertEqual("b", res.get_accent(2))

  def test_init_from_accents_with_pad_has_pad_at_idx_zero(self):
    res = AccentsDict.init_from_accents_with_pad({"b", "a"}, "xx")

    self.assertEqual("xx", res.get_accent(0))
    self.assertEqual("a", res.get_accent(1))
    self.assertEqual("b", res.get_accent(2))

  def test_init_from_accents_with_pad_ignores_existing_pad(self):
    res = AccentsDict.init_from_accents_with_pad({"b", "a", "xx"}, "xx")

    self.assertEqual("xx", res.get_accent(0))
    self.assertEqual("a", res.get_accent(1))
    self.assertEqual("b", res.get_accent(2))


if __name__ == '__main__':
  suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
  unittest.TextTestRunner(verbosity=2).run(suite)
