import unittest

from src.cli.utils import split_hparams_string


class UnitTests(unittest.TestCase):

  def test_split_hparams_string(self):
    hp_str = "test=123,abc=cde"

    res = split_hparams_string(hp_str)

    self.assertEqual(2, len(res))
    self.assertEqual("123", res["test"])
    self.assertEqual("cde", res["abc"])

  def test_split_hparams_string_none__returns_none(self):
    hp_str = None

    res = split_hparams_string(hp_str)

    self.assertIsNone(res)


if __name__ == '__main__':
  suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
  unittest.TextTestRunner(verbosity=2).run(suite)
