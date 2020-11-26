import sys
import time
import unittest

from src.core.common.text import *


class UnitTests(unittest.TestCase):

  def test_normalize_en(self):
    inp = "ü Hello my name is mr. test and    1 + 3 is $4. g 5e12  "
    res = normalize_en(inp)
    self.assertEqual(
      "u Hello my name is mister test and one + three is four dollars. grams five times ten to the power of twelve", res)

  def test_en_to_ipa(self):
    text = "This is a test. And an other one."
    start = time.time()
    res = []
    for _ in range(25):
      res.append(en_to_ipa(text, ENG_TO_IPA_MODE.EPITRAN))
    duration_s = time.time() - start
    # 21s with no caching
    self.assertTrue(duration_s < 6)
    self.assertEqual(25, len(res))
    self.assertEqual("ðɪs ɪz ə tɛst. ænd æn ʌðɹ̩ wʌn.", res[0])

  def test_ger_to_ipa(self):
    text = "Das ist ein Test. Und ein weiterer."
    start = time.time()
    res = []
    for _ in range(25):
      res.append(ger_to_ipa(text))
    duration_s = time.time() - start
    # 16.39s with no caching
    self.assertTrue(duration_s < 2)
    self.assertEqual(25, len(res))
    self.assertEqual("das ist ain test. und ain vaieteːrər.", res[0])

  def test_split_chn_text(self):
    example_text = "This is a test。 And an other one。\nAnd a new line。\r\nAnd a line with \r。\n\nAnd a line with \n in it。 This is a question？ This is a error！"

    res = split_chn_text(example_text)

    self.assertEqual(7, len(res))
    self.assertEqual("This is a test。", res[0])
    self.assertEqual("And an other one。", res[1])
    self.assertEqual("And a new line。", res[2])
    self.assertEqual("And a line with \r。", res[3])
    self.assertEqual("And a line with \n in it。", res[4])
    self.assertEqual("This is a question？", res[5])
    self.assertEqual("This is a error！", res[6])

  def test_split_ipa(self):
    example_text = "This is a test. And an other one.\nAnd a new line.\r\nAnd a line with \r.\n\nAnd a line with \n in it. This is a question? This is a error!"
    res = split_ipa_text(example_text)
    self.assertEqual(7, len(res))
    self.assertEqual("This is a test.", res[0])
    self.assertEqual("And an other one.", res[1])
    self.assertEqual("And a new line.", res[2])
    self.assertEqual("And a line with \r.", res[3])
    self.assertEqual("And a line with \n in it.", res[4])
    self.assertEqual("This is a question?", res[5])
    self.assertEqual("This is a error!", res[6])

  def test_split_en(self):
    example_text = "This is a test 4.000. And an other one.\nAnd a new line.\r\nAnd a line with \r.\n\nAnd a line with \n in it. This is a question? This is a error!"
    res = split_en_text(example_text)
    self.assertEqual(7, len(res))
    self.assertEqual("This is a test 4.000.", res[0])
    self.assertEqual("And an other one.", res[1])
    self.assertEqual("And a new line.", res[2])
    self.assertEqual("And a line with \r.", res[3])
    self.assertEqual("And a line with \n in it.", res[4])
    self.assertEqual("This is a question?", res[5])
    self.assertEqual("This is a error!", res[6])

  def test_split_en_north(self):
    example_text = "his cloak around him;\nand at last."
    res = split_en_text(example_text)
    self.assertEqual(2, len(res))
    self.assertEqual("his cloak around him;", res[0])
    self.assertEqual("and at last.", res[1])

  def test_split_ger(self):
    example_text = "Das ist ein Test 4.000. Und ein weiterer.\nUnd eine neue Zeile.\r\nUnd eine Zeile mit \r.\n\nUnd eine Zeile mit \n drin. Das ist eine Frage? Das ist ein Fehler!"
    res = split_ger_text(example_text)
    self.assertEqual(7, len(res))
    self.assertEqual("Das ist ein Test 4.000.", res[0])
    self.assertEqual("Und ein weiterer.", res[1])
    self.assertEqual("Und eine neue Zeile.", res[2])
    self.assertEqual("Und eine Zeile mit \r.", res[3])
    self.assertEqual("Und eine Zeile mit \n drin.", res[4])
    self.assertEqual("Das ist eine Frage?", res[5])
    self.assertEqual("Das ist ein Fehler!", res[6])

  def test_normal(self):
    inp = "东北军 的 一些 爱"

    res = chn_to_ipa(inp)

    self.assertEqual('tʊŋ˥peɪ˧˩˧tɕyn˥ tɤ i˥ɕjɛ˥ aɪ˥˩', res)

  def test_chn_to_ipa(self):
    text = "。？！，：；「」『』、"

    res = chn_to_ipa(text)

    self.assertEqual('.?!,:;"""",', res)

  # def test_period(self
  #   inp = "东北军 的 一些 爱"

  #   res = chn_to_ipa(inp)

  #   self.assertEqual('tʊŋ˥peɪ˧˩˧tɕyn˥ tɤ i˥ɕjɛ˥ aɪ˥˩。', res)

  # def test_question_mark_1(self):
  #   inp = "爱吗"

  #   res = chn_to_ipa(inp)

  #   self.assertEqual('aɪ˥˩ma？', res)

  # def test_question_mark_2(self):
  #   inp = "爱呢"

  #   res = chn_to_ipa(inp)

  #   self.assertEqual('aɪ˥˩nɤ？', res)

  # def test_line(self):
  #   inp = "东 一些"

  #   res = chn_to_ipa(inp)

  #   self.assertEqual('aɪ˥˩nɤ？', res)


if __name__ == '__main__':
  suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
  unittest.TextTestRunner(verbosity=2).run(suite)
