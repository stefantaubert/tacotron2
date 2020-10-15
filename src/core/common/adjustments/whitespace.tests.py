import unittest

from src.core.common.adjustments.whitespace import collapse_whitespace


class UnitTests(unittest.TestCase):

  def test_collapse_whitespace(self):
    res = collapse_whitespace("test  a b   c d  e \n  f")
    self.assertEqual("test a b c d e f", res)


if __name__ == '__main__':
  suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
  unittest.TextTestRunner(verbosity=2).run(suite)
