import unittest

from src.core.common.adjustments.numbers import (normalize_numbers,
                                                 replace_e_to_the_power_of,
                                                 replace_minus)


class UnitTests(unittest.TestCase):

  def test_replace_e_to_the_power_of__e_minus(self):
    res = replace_e_to_the_power_of("e-5654")
    self.assertEqual("ten to the power of minus 5654", res)

  def test_replace_e_to_the_power_of__e_no_minus(self):
    res = replace_e_to_the_power_of("e5654")
    self.assertEqual("ten to the power of 5654", res)

  def test_replace_e_to_the_power_of__prefix_e_no_minus(self):
    res = replace_e_to_the_power_of("45e5654")
    self.assertEqual("45 times ten to the power of 5654", res)

  def test_replace_e_to_the_power_of__prefix_e_minus(self):
    res = replace_e_to_the_power_of("45e-5654")
    self.assertEqual("45 times ten to the power of minus 5654", res)

  def test_replace_minus__normal(self):
    res = replace_minus("-5654")
    self.assertEqual("minus 5654", res)

  def test_replace_minus__with_e__no_replacement(self):
    res = replace_minus("e-5654")
    self.assertEqual("e-5654", res)

  def test_replace_minus__on_begin__replacement(self):
    res = replace_minus("-5654")
    self.assertEqual("minus 5654", res)

  def test_replace_minus__with_space__replacement(self):
    res = replace_minus(" -5654")
    self.assertEqual(" minus 5654", res)

  def test_normalize_numbers(self):
    res = normalize_numbers("$5654 -54 5e-21 test $300,000.40")
    self.assertEqual("five thousand, six hundred fifty-four dollars minus fifty-four five times ten to the power of minus twenty-one test three hundred thousand dollars, forty cents", res)


if __name__ == '__main__':
  suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
  unittest.TextTestRunner(verbosity=2).run(suite)
