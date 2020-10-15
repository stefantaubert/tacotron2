import unittest

from src.core.common.adjustments.emails import (replace_at_symbols,
                                                replace_mail_addresses)


class UnitTests(unittest.TestCase):

  def test_replace_mail_addresses_valid_mail(self):
    res = replace_mail_addresses("abc@def.de")
    self.assertEqual("abc at def dot de", res)

  def test_replace_mail_addresses_invalid_mail_no_dot(self):
    res = replace_mail_addresses("abc@defde")
    self.assertEqual("abc@defde", res)

  def test_replace_mail_addresses_invalid_mail_no_at(self):
    res = replace_mail_addresses("abcdef.de")
    self.assertEqual("abcdef.de", res)

  def test_replace_mail_addresses_invalid_mail_no_dot_and_at(self):
    res = replace_mail_addresses("abcdefde")
    self.assertEqual("abcdefde", res)

  def test_replace_at_symbols_no_space(self):
    res = replace_at_symbols("abc@def")
    self.assertEqual("abc at def", res)

  def test_replace_at_symbols_l_space(self):
    res = replace_at_symbols("abc @def")
    self.assertEqual("abc at def", res)

  def test_replace_at_symbols_r_space(self):
    res = replace_at_symbols("abc@ def")
    self.assertEqual("abc at def", res)

  def test_replace_at_symbols_both_space(self):
    res = replace_at_symbols("abc @ def")
    self.assertEqual("abc at def", res)

  def test_replace_at_symbols_double_at(self):
    res = replace_at_symbols("abc@@def")
    self.assertEqual("abc at  at def", res)


if __name__ == '__main__':
  suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
  unittest.TextTestRunner(verbosity=2).run(suite)
