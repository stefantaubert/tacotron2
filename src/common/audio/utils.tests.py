import numpy as np
import unittest
from src.common.audio.utils import float_to_wav, wav_to_float32

class UnitTests(unittest.TestCase):
  pass

if __name__ == '__main__':
  suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
  unittest.TextTestRunner(verbosity=2).run(suite)
