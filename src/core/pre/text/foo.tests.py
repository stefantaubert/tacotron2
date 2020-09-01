import unittest
import sys
from shutil import copyfile
from src.core.common import Language

from src.core.pre.text.foo import *

class UnitTests(unittest.TestCase):
  def test_split_eng(self):
    example_text = "This is a test. And an other one.\nAnd a new line.\r\nAnd a line with \r.\n\nAnd a line with \n in it. This is a question? This is a error!"
    res = split_sentences(example_text, Language.ENG)
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
