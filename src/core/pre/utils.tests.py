import unittest
from src.core.pre.wav.utils import get_chunk_name


class UnitTests(unittest.TestCase):
  
  def test_0_500_0_is_0_0(self):
    x = get_chunk_name(0, 500, 0)
    self.assertEqual("0-0", x)

  def test_0_500_1000_is_0_499(self):
    x = get_chunk_name(0, 500, 1000)
    self.assertEqual("0-499", x)

  def test_0_500_400_is_0_400(self):
    x = get_chunk_name(0, 500, 400)
    self.assertEqual("0-400", x)

  def test_500_500_1000_is_500_999(self):
    x = get_chunk_name(500, 500, 1000)
    self.assertEqual("500-999", x)

  def test_1000_500_1490_is_1000_1490(self):
    x = get_chunk_name(1000, 500, 1490)
    self.assertEqual("1000-1490", x)

if __name__ == '__main__':
  suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
  unittest.TextTestRunner(verbosity=2).run(suite)
