import unittest
from src.core.pre.wav.data import WavData


class UnitTests(unittest.TestCase):
  
  def test_repr(self):
    x = WavData(5, "", 0, 0)
    self.assertEqual("5", repr(x))

if __name__ == '__main__':
  suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
  unittest.TextTestRunner(verbosity=2).run(suite)
