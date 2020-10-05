import unittest

from src.core.tacotron.model import get_uniform_weights


class UnitTests(unittest.TestCase):

  def test_get_uniform_weights(self):
    res = get_uniform_weights(5, 100)
    self.assertNotEqual(0, res[0][0])
    self.assertEqual(5, res.shape[0])
    self.assertEqual(100, res.shape[1])


if __name__ == '__main__':
  suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
  unittest.TextTestRunner(verbosity=2).run(suite)
