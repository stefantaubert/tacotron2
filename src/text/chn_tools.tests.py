import unittest
import sys

sys.path.append('../tacotron2')

from src.text.chn_tools import *

class UnitTests(unittest.TestCase):
    
  def test_normal(self):
    inp = "东北军 的 一些 爱"

    res = chn_to_ipa(inp)

    self.assertEqual('tʊŋ˥peɪ˧˩˧tɕyn˥ tɤ i˥ɕjɛ˥ aɪ˥˩', res)

  def test_period(self):
    inp = "东北军 的 一些 爱"

    res = chn_to_ipa(inp, add_period=True)

    self.assertEqual('tʊŋ˥peɪ˧˩˧tɕyn˥ tɤ i˥ɕjɛ˥ aɪ˥˩.', res)

  def test_question_mark_1(self):
    inp = "爱吗"

    res = chn_to_ipa(inp)

    self.assertEqual('aɪ˥˩ma?', res)

  def test_question_mark_2(self):
    inp = "爱呢"

    res = chn_to_ipa(inp)

    self.assertEqual('aɪ˥˩nɤ?', res)

  def test_line(self):
    inp = "东 一些"

    res = chn_to_ipa(inp)

    self.assertEqual('aɪ˥˩nɤ?', res)

if __name__ == '__main__':
  suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
  unittest.TextTestRunner(verbosity=2).run(suite)
