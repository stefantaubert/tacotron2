import unittest

import torch

from src.core.common.mel_plot import concatenate_mels_core


class UnitTests(unittest.TestCase):
  
  def test_concatenate_cpu_no_pause(self):
    a = torch.rand((71, 32, 1), device="cpu")
    b = torch.zeros((29, 32, 1), device="cpu")
    x = concatenate_mels_core([a, b])

    self.assertEqual(100, len(x))

  def test_concatenate_gpu(self):
    a = torch.rand((71, 32, 1), device="cuda")
    b = torch.zeros((29, 32, 1), device="cuda")
    x = concatenate_mels_core([a, b])

    self.assertEqual(100, len(x))

  def test_concatenate_cpu_one_pause(self):
    a = torch.rand((71, 32, 1), device="cpu")
    b = torch.zeros((29, 32, 1), device="cpu")
    x = concatenate_mels_core([a, b], sentence_pause_samples_count=50)

    self.assertEqual(100 + 50, len(x))

  def test_concatenate_cpu_two_pause(self):
    a = torch.rand((71, 32, 1), device="cpu")
    b = torch.zeros((29, 32, 1), device="cpu")
    c = torch.zeros((30, 32, 1), device="cpu")
    x = concatenate_mels_core([a, b, c], sentence_pause_samples_count=50)

    self.assertEqual(130 + 100, len(x))

  def test_concatenate_cpu_one_element(self):
    a = torch.rand((100, 32, 1), device="cpu")
    x = concatenate_mels_core([a], sentence_pause_samples_count=50)

    self.assertEqual(100, len(x))

if __name__ == '__main__':
  suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
  unittest.TextTestRunner(verbosity=2).run(suite)
