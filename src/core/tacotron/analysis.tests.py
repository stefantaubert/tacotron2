import logging
import numpy as np
import unittest
from src.core.tacotron.analysis import norm2emb, get_similarities, sims_to_csv, emb_plot_2d, emb_plot_3d, plot_embeddings
from src.core.pre import SymbolIdDict
import torch

class UnitTests(unittest.TestCase):
  def test_norm2emb(self):
    emb = torch.ones(size=(2,3))

    res = norm2emb(emb)

    self.assertEqual(len(res), len(emb))

  def test_get_similarities_one(self):
    emb = torch.ones(size=(2,3))

    res = get_similarities(emb)

    self.assertEqual(len(res), len(emb))
    self.assertEqual(res[0], [(1, 1)])
    self.assertEqual(res[1], [(0, 1)])

  def test_get_similarities_zero(self):
    emb = torch.ones(size=(2,3))
    torch.nn.init.zeros_(emb[0])

    res = get_similarities(emb)

    self.assertEqual(len(res), len(emb))
    self.assertTrue(np.isnan(res[0][0][1]))
    self.assertTrue(np.isnan(res[1][0][1]))

  def test_get_similarities_is_sorted(self):
    symbols = SymbolIdDict.init_from_symbols({"a"})
    emb = np.zeros(shape=(3,3))
    emb[symbols.get_id("_")] = [0.5, 1.0, 0]
    emb[symbols.get_id("~")] = [1.0, 0.6, 0]
    emb[symbols.get_id("a")] = [1.0, 0.5, 0]

    sims = get_similarities(emb)

    self.assertEqual(3, len(sims))
    self.assertEqual(symbols.get_id("~"), sims[symbols.get_id("_")][0][0])
    self.assertEqual(symbols.get_id("a"), sims[symbols.get_id("_")][1][0])
    self.assertEqual(symbols.get_id("~"), sims[symbols.get_id("a")][0][0])
    self.assertEqual(symbols.get_id("_"), sims[symbols.get_id("a")][1][0])
    self.assertEqual(symbols.get_id("a"), sims[symbols.get_id("~")][0][0])
    self.assertEqual(symbols.get_id("_"), sims[symbols.get_id("~")][1][0])

  def test_sims_to_csv(self):
    emb = torch.ones(size=(2,3))
    torch.nn.init.zeros_(emb[0])
    sims = get_similarities(emb)
    symbols = SymbolIdDict.init_from_symbols({})
    res = sims_to_csv(sims, symbols)
    self.assertEqual(2, len(res.index))
    self.assertListEqual(['_', '<=>', '~', 'nan'], list(res.values[0]))
    self.assertListEqual(['~', '<=>', '_', 'nan'], list(res.values[1]))

  def test_emb_plot_2d(self):
    emb = torch.ones(size=(2,3))
    symbols = ["_", "~"]
    res = emb_plot_2d(emb, symbols)
    self.assertEqual("2D-Embeddings", res.layout.title.text)

  def test_emb_plot_3d(self):
    emb = torch.ones(size=(2,3))
    symbols = ["_", "~"]
    res = emb_plot_3d(emb, symbols)
    self.assertEqual("3D-Embeddings", res.layout.title.text)

  def test_plot_embeddings(self):
    emb = torch.ones(size=(2,3))
    symbols = SymbolIdDict.init_from_symbols({})
    text, plot2d, plot3d = plot_embeddings(symbols, emb, logging.getLogger())

    self.assertEqual(2, len(text.index))
    self.assertListEqual(['_', '<=>', '~', '1.00'], list(text.values[0]))
    self.assertListEqual(['~', '<=>', '_', '1.00'], list(text.values[1]))
    self.assertEqual("2D-Embeddings", plot2d.layout.title.text)
    self.assertEqual("3D-Embeddings", plot3d.layout.title.text)

if __name__ == '__main__':
  suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
  unittest.TextTestRunner(verbosity=2).run(suite)
