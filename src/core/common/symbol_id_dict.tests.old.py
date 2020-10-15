# import sys
# import unittest
# from shutil import copyfile

# from src.core.common import *

# path_to_v1 = '/tmp/symbols.json'
# path_to_v2 = '/tmp/dumpv2.json'
# path_to_plot = '/tmp/plot.txt'

# class UnitTests(unittest.TestCase):

#   def test_init_from_symbols(self):
#     symbols = {'a', 'c', 'b'}
#     c = init_from_symbols(symbols)
#     internal_symbols_count = 2 # pad and eos
#     self.assertEqual(3 + internal_symbols_count, c.get_symbols_count())

#   def test_init_from_symbols_keys_are_int(self):
#     c = init_from_symbols({'a', 'c', 'b'})
#     keys = c.get_symbol_ids()
#     self.assertIs(int, type(keys[0]))

#   def test_load_v2_keys_are_int(self):
#     c = init_from_symbols({'a', 'c', 'b'})
#     c.dump(path_to_v2)
#     res = load_from_file_v2(path_to_v2)
#     keys = c.get_symbol_ids()
#     self.assertIs(int, type(keys[0]))

#   def test_load_v1_keys_are_int(self):
#     res = load_from_file_v1(path_to_v1)
#     keys = res.get_symbol_ids()
#     self.assertIs(int, type(keys[0]))

#   def test_symbols_to_ids(self):
#     symbols = {'a', 'b', 'c'}
#     c = init_from_symbols(symbols)

#     res = c.symbols_to_ids(['a', 'b', 'c', 'a', 'a', 'x'], add_eos=False, replace_unknown_with_pad=False)

#     self.assertEqual([2, 3, 4, 2, 2], res)

#   def test_get_unknown_symbols(self):
#     symbols = {'a', 'b', 'c'}
#     c = init_from_symbols(symbols)

#     res = c.get_unknown_symbols(['a', 'b', 'c', 'a', 'y', 'x'])

#     self.assertEqual({'y', 'x'}, res)

#   def test_symbols_to_ids_with_multiple_ids_per_symbol(self):
#     symbols = {'a', 'b', 'c'}
#     c = init_from_symbols(symbols)
#     c.add_symbols({'a', 'b', 'c'}, False, subset_id=1)

#     res = c.symbols_to_ids(['a', 'b', 'c'], add_eos=False, replace_unknown_with_pad=False, subset_id_if_multiple=0)

#     self.assertEqual([2, 3, 4], res)

#     res = c.symbols_to_ids(['a', 'b', 'c'], add_eos=False, replace_unknown_with_pad=False, subset_id_if_multiple=1)

#     self.assertEqual([7, 8 ,9], res)

#   def test_symbols_to_ids_replace_with_pad(self):
#     symbols = {'a', 'b', 'c'}
#     c = init_from_symbols(symbols)

#     res = c.symbols_to_ids(['x', 'x', 'x'], add_eos=False, replace_unknown_with_pad=True)

#     self.assertEqual(3, len(res))

#   def test_ids_to_symbols(self):
#     symbols = {'a', 'b', 'c'}
#     c = init_from_symbols(symbols)

#     res = c.ids_to_symbols([2, 3, 4, 2, 2])

#     self.assertEqual(['a', 'b', 'c', 'a', 'a'], res)

#   def test_symbols_to_ids_after_adding_using_new_eos_and_pad(self):
#     symbols = {'a', 'b', 'c'}
#     c = init_from_symbols(symbols)
#     c.add_symbols(symbols, ignore_existing=False, subset_id=1)
#     res = c.symbols_to_ids(['a', 'b', 'x'], subset_id_if_multiple=1, add_eos=True, replace_unknown_with_pad=True)

#     self.assertEqual([7, 8, 5, 6], res)

#   def test_ids_to_text(self):
#     symbols = {'a', 'b', 'c'}
#     c = init_from_symbols(symbols)

#     res = c.get_text([2, 3, 4, 2, 2])

#     self.assertEqual('abcaa', res)

#   def test_plot(self):
#     symbols = {'a', 'c', 'b'}
#     c = init_from_symbols(symbols)
#     c.plot(path_to_plot)
#     with open(path_to_plot, 'r', encoding='utf-8') as f:
#       lines = f.readlines()
#     lines = [x.rstrip() for x in lines]

#     self.assertEqual(['0\t_\t0', '0\t~\t1', '0\ta\t2', '0\tb\t3', '0\tc\t4'], lines)

#   def test_load(self):
#     symbols = {'a', 'c', 'b'}
#     c = init_from_symbols(symbols)
#     c.dump(path_to_v2)
#     res = load_from_file_v2(path_to_v2)
#     symbols = res.get_symbols(include_subset_id=False, include_id=False)
#     self.assertEqual(['_', '~', 'a', 'b', 'c'], symbols)

#   def test_load_v1(self):
#     res = load_from_file_v1(path_to_v1)
#     symbols = res.get_symbols(include_subset_id=False, include_id=False)
#     self.assertEqual(' ', symbols[0])
#     self.assertEqual('!', symbols[1])
#     self.assertEqual("\u0265", symbols[-1])
#     self.assertEqual(79, len(symbols))

#   def test_load_v1_conversion(self):
#     backup_path = "{}.backup".format(path_to_v1)
#     copyfile(path_to_v1, backup_path)
#     res = load_from_file(path_to_v1, convert_v1_to_v2=True)
#     try:
#       res = load_from_file_v2(path_to_v1)
#       symbols = res.get_symbols(include_subset_id=False, include_id=False)
#       self.assertEqual(79, len(symbols))
#     except:
#       self.fail()
#     finally:
#       copyfile(backup_path, path_to_v1)

#   def test_load_version_detection(self):
#     res = load_from_file(path_to_v1, convert_v1_to_v2=False)
#     symbols = res.get_symbols(include_subset_id=False, include_id=False)
#     self.assertEqual(79, len(symbols))

#     symbols = {'a', 'c', 'b'}
#     c = init_from_symbols(symbols)
#     c.dump(path_to_v2)
#     res = load_from_file(path_to_v2)
#     symbols = res.get_symbols(include_subset_id=False, include_id=False)
#     self.assertEqual(5, len(symbols))

#   def test_add(self):
#     symbols = {'a', 'c', 'b'}
#     c = init_from_symbols(symbols)
#     c.add_symbols({'x', 'a'}, ignore_existing=True, subset_id=1)
#     symbols = c.get_symbols(include_subset_id=False, include_id=False)
#     self.assertEqual(['_', '~', 'a', 'b', 'c', 'x'], symbols)

#   def test_load_saves_order(self):
#     symbols = {'a', 'c', 'b'}
#     c = init_from_symbols(symbols)
#     c.add_symbols({'a', 'x'}, ignore_existing=False, subset_id=1)
#     c.dump(path_to_v2)
#     res = load_from_file_v2(path_to_v2)
#     symbols = res.get_symbols(include_subset_id=False, include_id=False)
#     self.assertEqual(['_', '~', 'a', 'b', 'c', '_', '~', 'a', 'x'], symbols)

# if __name__ == '__main__':
#   suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
#   unittest.TextTestRunner(verbosity=2).run(suite)
