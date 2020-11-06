import unittest

from src.core.pre.parser.custom import parse


class UnitTests(unittest.TestCase):
  def test_all_utterances_are_included(self):
    dest = '/datasets/NNLV_pilot'
    res = parse(dest)
    self.assertEqual(len(res), 188)

if __name__ == '__main__':

  suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
  unittest.TextTestRunner(verbosity=2).run(suite)
