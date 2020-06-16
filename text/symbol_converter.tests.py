import unittest
import sys

sys.path.append('../tacotron2')

#from SymbolConverter import SymbolConverter
from text.symbol_converter import *

class UnitTests(unittest.TestCase):
    
  def test_init_from_symbols(self):
    symbols = {'a', 'c', 'b'}
    c = init_from_symbols(symbols)
    internal_symbols_count = 2 # pad and eos
    self.assertEqual(3 + internal_symbols_count, c.get_symbol_ids_count())

  def test_symbols_to_ids(self):
    symbols = {'a', 'b', 'c'}
    c = init_from_symbols(symbols)
    
    res = c.symbols_to_ids(['a', 'b', 'c', 'a', 'a', 'x'], add_eos=False, replace_unknown_with_pad=False)

    self.assertEqual([2, 3, 4, 2, 2], list(res))

  def test_symbols_to_ids_with_multiple_ids_per_symbol(self):
    symbols = {'a', 'b', 'c'}
    c = init_from_symbols(symbols)
    c.add_symbols({'a', 'b', 'c'}, False, subset_id=1)
    
    res = c.symbols_to_ids(['a', 'b', 'c'], add_eos=False, replace_unknown_with_pad=False, subset_id_if_multiple=0)

    self.assertEqual([2, 3, 4], list(res))

    res = c.symbols_to_ids(['a', 'b', 'c'], add_eos=False, replace_unknown_with_pad=False, subset_id_if_multiple=1)

    self.assertEqual([5, 6, 7], list(res))

  def test_symbols_to_ids_replace_with_pad(self):
    symbols = {'a', 'b', 'c'}
    c = init_from_symbols(symbols)
    
    res = c.symbols_to_ids(['x', 'x', 'x'], add_eos=False, replace_unknown_with_pad=True)

    self.assertEqual(3, len(res))

  def test_ids_to_symbols(self):
    symbols = {'a', 'b', 'c'}
    c = init_from_symbols(symbols)
    
    res = c.ids_to_symbols([2, 3, 4, 2, 2])

    self.assertEqual(['a', 'b', 'c', 'a', 'a'], res)

  def test_ids_to_text(self):
    symbols = {'a', 'b', 'c'}
    c = init_from_symbols(symbols)
    
    res = c.ids_to_text([2, 3, 4, 2, 2])

    self.assertEqual('abcaa', res)

  def test_plot(self):
    symbols = {'a', 'c', 'b'}
    c = init_from_symbols(symbols)
    path = '/tmp/plotted.txt'
    c.plot(path, sort=False)
    with open(path, 'r', encoding='utf-8') as f:
      lines = f.readlines()
    lines = [x.rstrip() for x in lines]

    self.assertEqual(['0\t_\t0', '0\t~\t1', '0\ta\t2', '0\tb\t3', '0\tc\t4'], lines)

  def test_load(self):
    symbols = {'a', 'c', 'b'}
    c = init_from_symbols(symbols)
    path = '/tmp/plotted.txt'
    c.dump(path)
    res = load_from_file_v2(path)
    symbols = res.get_symbols(include_subset_id=False, include_id=False)
    self.assertEqual(['_', '~', 'a', 'b', 'c'], symbols)

  def test_load_v1(self):
    path = '/tmp/symbols.json'
    res = load_from_file_v1(path)
    symbols = res.get_symbols(include_subset_id=False, include_id=False)
    self.assertEqual(' ', symbols[0])
    self.assertEqual('!', symbols[1])
    self.assertEqual("\u0265", symbols[-1])
    self.assertEqual(79, len(symbols))

  def test_load_version_detection(self):
    path = '/tmp/symbols.json'
    res = load_from_file(path)
    symbols = res.get_symbols(include_subset_id=False, include_id=False)
    self.assertEqual(79, len(symbols))

    path = '/tmp/dumped.json'
    symbols = {'a', 'c', 'b'}
    c = init_from_symbols(symbols)
    c.dump(path)
    res = load_from_file(path)
    symbols = res.get_symbols(include_subset_id=False, include_id=False)
    self.assertEqual(5, len(symbols))

  def test_add(self):
    symbols = {'a', 'c', 'b'}
    c = init_from_symbols(symbols)
    c.add_symbols({'x', 'a'}, ignore_existing=True, subset_id=1)
    symbols = c.get_symbols(include_subset_id=False, include_id=False)
    self.assertEqual(['_', '~', 'a', 'b', 'c', 'x'], symbols)

  def test_load_saves_order(self):
    symbols = {'a', 'c', 'b'}
    c = init_from_symbols(symbols)
    c.add_symbols({'a', 'x'}, ignore_existing=False, subset_id=1)
    path = '/tmp/dumped.json'
    c.dump(path)
    res = load_from_file_v2(path)
    symbols = res.get_symbols(include_subset_id=False, include_id=False)
    self.assertEqual(['_', '~', 'a', 'b', 'c', 'a', 'x'], symbols)

if __name__ == '__main__':
  suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
  unittest.TextTestRunner(verbosity=2).run(suite)
