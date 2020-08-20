import unittest
from src.core.tacotron.training import load_symbol_embedding_weights_from, get_uniform_weights
from src.core.pre import SymbolConverter
from torch import nn

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

if __name__ == '__main__':
  suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
  unittest.TextTestRunner(verbosity=2).run(suite)
