import unittest
import sys

from src.core.pre.text.utils import *

class UnitTests(unittest.TestCase):
  def test_split_chn_text(self):
    example_text = "This is a test。 And an other one。\nAnd a new line。\r\nAnd a line with \r。\n\nAnd a line with \n in it。 This is a question？ This is a error！"

    res = split_chn_text(example_text)
    
    self.assertEqual(7, len(res))
    self.assertEqual("This is a test.", res[0])
    self.assertEqual("And an other one.", res[1])
    self.assertEqual("And a new line.", res[2])
    self.assertEqual("And a line with \r.", res[3])
    self.assertEqual("And a line with \n in it.", res[4])
    self.assertEqual("This is a question?", res[5])
    self.assertEqual("This is a error!", res[6])

if __name__ == '__main__':
  suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
  unittest.TextTestRunner(verbosity=2).run(suite)
