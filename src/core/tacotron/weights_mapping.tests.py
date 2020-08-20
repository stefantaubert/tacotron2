import os
import unittest
from src.core.tacotron.weights_mapping import get_mapped_embedding_weights, SymbolsMap
from src.core.pre import SymbolConverter
from torch import nn
import tempfile

class UnitTests(unittest.TestCase):
  
  def test_from_two_sets(self):
    m = SymbolsMap.from_two_sets({"a", "b"}, {"b", "c"})

    self.assertEqual(2, len(m))
    self.assertEqual("b", m["b"])
    self.assertEqual("", m["c"])

  def test_save_load_symbols_map(self):
    path = tempfile.mktemp()
    symbols_map = SymbolsMap([
      ("b", "a"),
      ("c", "b"),
      ("x", "y"),
    ])
    symbols_map.save(path)
    res = SymbolsMap.load(path)
    os.remove(path)
    self.assertEqual(3, len(res))
    self.assertEqual("a", res["b"])
    self.assertEqual("b", res["c"])
    self.assertEqual("y", res["x"])

  def test_get_mapped_embedding_weights_no_map(self):
    model_conv = SymbolConverter.init_from_symbols({"b", "c"})
    model_embeddings = nn.Embedding(model_conv.get_symbol_ids_count(), 1)
    nn.init.zeros_(model_embeddings.weight)

    trained_symbols = SymbolConverter.init_from_symbols({"a", "b"})
    trained_embeddings = nn.Embedding(trained_symbols.get_symbol_ids_count(), 1)
    trained_embeddings.weight[trained_symbols.symbol_to_id("~")] = 1
    trained_embeddings.weight[trained_symbols.symbol_to_id("_")] = 2
    trained_embeddings.weight[trained_symbols.symbol_to_id("a")] = 3
    trained_embeddings.weight[trained_symbols.symbol_to_id("b")] = 4

    mapped_emb_weights = get_mapped_embedding_weights(
      model_weights=model_embeddings.weight,
      model_symbols=model_conv,
      trained_weights=trained_embeddings.weight,
      trained_symbols=trained_symbols,
    )

    self.assertEqual(1, mapped_emb_weights[model_conv.symbol_to_id("~")][0].item())
    self.assertEqual(2, mapped_emb_weights[model_conv.symbol_to_id("_")][0].item())
    self.assertEqual(4, mapped_emb_weights[model_conv.symbol_to_id("b")][0].item())
    self.assertEqual(0, mapped_emb_weights[model_conv.symbol_to_id("c")][0].item())

  def test_get_mapped_embedding_weights_with_map(self):
    model_conv = SymbolConverter.init_from_symbols({"b", "c", "d"})
    model_embeddings = nn.Embedding(model_conv.get_symbol_ids_count(), 1)
    nn.init.zeros_(model_embeddings.weight)

    trained_symbols = SymbolConverter.init_from_symbols({"a", "b"})
    trained_embeddings = nn.Embedding(trained_symbols.get_symbol_ids_count(), 1)
#    nn.init.ones_(trained_embeddings.weight)
    trained_embeddings.weight[trained_symbols.symbol_to_id("a")] = 1
    trained_embeddings.weight[trained_symbols.symbol_to_id("b")] = 2
    symbols_map = SymbolsMap([
      ("b", "a"),
      ("c", "b"),
      ("x", "y"),
    ])

    mapped_emb_weights = get_mapped_embedding_weights(
      model_weights=model_embeddings.weight,
      model_symbols=model_conv,
      trained_weights=trained_embeddings.weight,
      trained_symbols=trained_symbols,
      symbols_map=symbols_map
    )

    self.assertEqual(0, mapped_emb_weights[model_conv.symbol_to_id("~")][0].item())
    self.assertEqual(0, mapped_emb_weights[model_conv.symbol_to_id("_")][0].item())
    self.assertEqual(1, mapped_emb_weights[model_conv.symbol_to_id("b")][0].item())
    self.assertEqual(2, mapped_emb_weights[model_conv.symbol_to_id("c")][0].item())
    self.assertEqual(0, mapped_emb_weights[model_conv.symbol_to_id("d")][0].item())

if __name__ == '__main__':
  suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
  unittest.TextTestRunner(verbosity=2).run(suite)
