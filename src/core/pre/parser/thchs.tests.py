import numpy as np
import unittest
from src.core.pre.parser.thchs import parse

class UnitTests(unittest.TestCase):
  def test_ordered(self):
    dest = '/datasets/thchs_wav'
    res = parse(dest)
    speakers = []
    data = {}
    for x in res:
      if x[1] not in speakers:
        speakers.append(x[1])
        data[x[1]] = []
      data[x[1]].append(x[0])
    
    self.assertEqual(data["B15"][0], "B15_250")
    self.assertEqual(data["B15"][-1], "B15_499")
    self.assertEqual(speakers[0], "A2")
    self.assertEqual(speakers[-1], "D32")
    self.assertEqual(len(speakers), 60)

  def test_all_utterances_are_included(self):
      dest = '/datasets/thchs_wav'
      res = parse(dest)
      self.assertEqual(len(res), 12495)

  def test_contains_only_name_speaker_text_path_lang(self):
      dest = '/datasets/thchs_wav'
      res = parse(dest)
      for x in res:
        self.assertEqual(len(x), 5)

if __name__ == '__main__':
  suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
  unittest.TextTestRunner(verbosity=2).run(suite)
