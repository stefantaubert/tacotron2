import os
import shutil
import unittest
from os import remove, removedirs
from tempfile import mkdtemp

import numpy as np

from src.core.common.utils import (cosine_dist_mels, get_basename,
                                   get_chunk_name, get_subfolders,
                                   make_same_dim, pass_lines)


class UnitTests(unittest.TestCase):

  def test_pass_lines(self):
    text = "abc\ndef\ngh"
    res = []
    pass_lines(res.append, text)
    self.assertEqual(["abc", "def", "gh"], res)

  def test_get_basename_full_path(self):
    path = "/a/b/c/test.wav.xyz"

    res = get_basename(path)

    self.assertEqual("test.wav", res)

  def test_get_basename_of_filename(self):
    path = "test.wav.xyz"

    res = get_basename(path)

    self.assertEqual("test.wav", res)

  def test_get_basename_of_dir_wo_slash(self):
    path = "/a/b/c/test"

    res = get_basename(path)

    self.assertEqual("test", res)

  def test_get_basename_of_dir_w_slash(self):
    path = "/a/b/c/test/"

    res = get_basename(path)

    self.assertEqual("", res)

  def test_get_subfolders(self):
    testdir = mkdtemp()
    test1 = os.path.join(testdir, "test1")
    test2 = os.path.join(testdir, "test2")
    os.makedirs(test1)
    os.makedirs(test2)

    res = get_subfolders(testdir)
    shutil.rmtree(testdir)

    self.assertEqual([test1, test2], res)

  def test_cosine_dist_mels_1_minus1(self):
    a = np.ones(shape=(80, 1))
    b = np.ones(shape=(80, 1))
    score = cosine_dist_mels(a, -b)
    self.assertEqual(-1, score)

  def test_cosine_dist_mels_1_0(self):
    a = np.ones(shape=(80, 1))
    b = np.ones(shape=(80, 0))
    score = cosine_dist_mels(a, b)
    self.assertEqual(0, score)

  def test_cosine_dist_mels_1_1(self):
    a = np.ones(shape=(80, 1))
    b = np.ones(shape=(80, 1))
    score = cosine_dist_mels(a, b)
    self.assertEqual(1, score)

  def test_make_same_dim_1_0(self):
    a = np.ones(shape=(80, 1))
    b = np.ones(shape=(80, 0))
    a_res, b_res = make_same_dim(a, b)
    self.assertEqual((80, 1), a_res.shape)
    self.assertEqual((80, 1), b_res.shape)
    self.assertEqual(1, a_res[0])
    self.assertEqual(0, b_res[0])

  def test_make_same_dim_1_1(self):
    a = np.ones(shape=(80, 1))
    b = np.ones(shape=(80, 1))
    a_res, b_res = make_same_dim(a, b)
    self.assertEqual((80, 1), a_res.shape)
    self.assertEqual((80, 1), b_res.shape)
    self.assertEqual(1, a_res[0])
    self.assertEqual(1, b_res[0])

  def test_make_same_dim_0_1(self):
    a = np.ones(shape=(80, 0))
    b = np.ones(shape=(80, 1))
    a_res, b_res = make_same_dim(a, b)
    self.assertEqual((80, 1), a_res.shape)
    self.assertEqual((80, 1), b_res.shape)
    self.assertEqual(0, a_res[0])
    self.assertEqual(1, b_res[0])

  def test_0_500_0_is_0_0(self):
    x = get_chunk_name(0, 500, 0)
    self.assertEqual("0-0", x)

  def test_0_500_1000_is_0_499(self):
    x = get_chunk_name(0, 500, 1000)
    self.assertEqual("0-499", x)

  def test_0_500_400_is_0_400(self):
    x = get_chunk_name(0, 500, 400)
    self.assertEqual("0-400", x)

  def test_500_500_1000_is_500_999(self):
    x = get_chunk_name(500, 500, 1000)
    self.assertEqual("500-999", x)

  def test_1000_500_1490_is_1000_1490(self):
    x = get_chunk_name(1000, 500, 1490)
    self.assertEqual("1000-1490", x)


if __name__ == '__main__':
  suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
  unittest.TextTestRunner(verbosity=2).run(suite)
