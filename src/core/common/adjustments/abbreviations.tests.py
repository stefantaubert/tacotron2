import unittest

from src.core.common.adjustments.abbreviations import (
    expand_abbreviations, expand_units_of_measure,
    replace_big_letter_abbreviations)


class UnitTests(unittest.TestCase):

  def test_replace_big_letter_abbreviations(self):
    res = replace_big_letter_abbreviations("BC BC BCab abBC BC")
    self.assertEqual("B C B C B Cab abB C B C", res)

  def test_expand_abbreviations(self):
    res = expand_abbreviations(
      "mrs. mr. dr. st. co. jr. maj. gen. drs. rev. lt. hon. sgt. capt. esq. ltd. col. ft. Mrs. mrs gen")
    self.assertEqual(
      "misess mister doctor saint company junior major general doctors reverend lieutenant honorable sergeant captain esquire limited colonel fort misess mrs gen", res)

  def test_expand_units_of_measure(self):
    res = expand_units_of_measure("g kg mm cm m S s he's min.")
    self.assertEqual("g kilograms millimeters centimeters meters S seconds he's minutes.", res)


if __name__ == '__main__':
  suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
  unittest.TextTestRunner(verbosity=2).run(suite)
