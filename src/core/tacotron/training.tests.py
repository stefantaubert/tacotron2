import unittest

from torch import nn

from src.core.common.symbol_id_dict import SymbolIdDict
from src.core.common.symbols_map import SymbolsMap
from src.core.tacotron.training import (get_mapped_embedding_weights_core,
                                        get_uniform_weights,
                                        load_symbol_embedding_weights_from)


class UnitTests(unittest.TestCase):
  def test_load_symbol_emb_weights_from(self):
    model_path = "/datasets/models/taco2pt_v2/ljs_ipa_ms_from_scratch/checkpoints/113500"
    x = load_symbol_embedding_weights_from(model_path)
    self.assertEqual(512,x.shape[1])

  def test_get_uniform_weights(self):
    res = get_uniform_weights(5, 100)
    self.assertNotEqual(0, res[0][0])
    self.assertEqual(5, res.shape[0])
    self.assertEqual(100, res.shape[1])

  def test_get_mapped_embedding_weights_no_map(self):
    model_conv = SymbolIdDict.init_from_symbols({"1", "2", "b", "c"})
    model_embeddings = nn.Embedding(len(model_conv), 1)
    nn.init.zeros_(model_embeddings.weight)

    trained_symbols = SymbolIdDict.init_from_symbols({"1", "2", "a", "b"})
    trained_embeddings = nn.Embedding(len(trained_symbols), 1)
    trained_embeddings.weight[trained_symbols.get_id("1")] = 1
    trained_embeddings.weight[trained_symbols.get_id("2")] = 2
    trained_embeddings.weight[trained_symbols.get_id("a")] = 3
    trained_embeddings.weight[trained_symbols.get_id("b")] = 4

    mapped_emb_weights = get_mapped_embedding_weights_core(
      model_weights=model_embeddings.weight,
      model_symbols=model_conv,
      trained_weights=trained_embeddings.weight,
      trained_symbols=trained_symbols,
    )

    self.assertEqual(1, mapped_emb_weights[model_conv.get_id("1")][0].item())
    self.assertEqual(2, mapped_emb_weights[model_conv.get_id("2")][0].item())
    self.assertEqual(4, mapped_emb_weights[model_conv.get_id("b")][0].item())
    self.assertEqual(0, mapped_emb_weights[model_conv.get_id("c")][0].item())

  def test_get_mapped_embedding_weights_with_map(self):
    model_conv = SymbolIdDict.init_from_symbols({"1", "2", "b", "c", "d"})
    model_embeddings = nn.Embedding(len(model_conv), 1)
    nn.init.zeros_(model_embeddings.weight)

    trained_symbols = SymbolIdDict.init_from_symbols({"1", "2", "a", "b"})
    trained_embeddings = nn.Embedding(len(trained_symbols), 1)
    trained_embeddings.weight[trained_symbols.get_id("a")] = 1
    trained_embeddings.weight[trained_symbols.get_id("b")] = 2
    symbols_map = SymbolsMap([
      ("b", "a"),
      ("c", "b"),
      ("x", "y"),
    ])

    mapped_emb_weights = get_mapped_embedding_weights_core(
      model_weights=model_embeddings.weight,
      model_symbols=model_conv,
      trained_weights=trained_embeddings.weight,
      trained_symbols=trained_symbols,
      symbols_mapping=symbols_map
    )

    self.assertEqual(0, mapped_emb_weights[model_conv.get_id("1")][0].item())
    self.assertEqual(0, mapped_emb_weights[model_conv.get_id("2")][0].item())
    self.assertEqual(1, mapped_emb_weights[model_conv.get_id("b")][0].item())
    self.assertEqual(2, mapped_emb_weights[model_conv.get_id("c")][0].item())
    self.assertEqual(0, mapped_emb_weights[model_conv.get_id("d")][0].item())

if __name__ == '__main__':
  suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
  unittest.TextTestRunner(verbosity=2).run(suite)
