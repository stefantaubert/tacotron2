import unittest
from dataclasses import dataclass
from typing import List

from torch.utils.data import DataLoader, Dataset

from src.core.common.train import (check_is_first, check_is_last,
                                   check_is_last_batch_iteration,
                                   check_is_save_epoch,
                                   check_is_save_iteration,
                                   get_continue_batch_iteration,
                                   get_continue_epoch, get_dataclass_from_dict,
                                   get_only_known_params, get_value_in_type,
                                   iteration_to_batch_iteration,
                                   iteration_to_epoch, skip_batch)


class DummyDataset(Dataset):
  def __init__(self, items: List[int]) -> None:
    self.items = items

  def __getitem__(self, index: int) -> int:
    return self.items[index]

  def __len__(self) -> int:
    return len(self.items)


class UnitTests(unittest.TestCase):
  def test_get_dataclass_from_dict(self):
    @dataclass
    class DummyHp:
      b: int = 4

    inp = {"a": "1", "b": "2"}
    res, ignored = get_dataclass_from_dict(inp, DummyHp)
    self.assertIsInstance(res, DummyHp)
    self.assertEqual({"a"}, ignored)

  def test_get_only_known_params(self):
    @dataclass
    class DummyHp:
      b: int = 4

    inp = {"a": "1", "b": "2"}

    res = get_only_known_params(inp, DummyHp())

    self.assertEqual(1, len(res))
    self.assertEqual("2", res["b"])

  def test_get_value_in_type_int__returns_int(self):
    res = get_value_in_type(1, "2")
    self.assertTrue(isinstance(res, int))
    self.assertEqual(2, res)

  def test_get_value_in_type_float__returns_float(self):
    res = get_value_in_type(1.0, "2")
    self.assertTrue(isinstance(res, float))
    self.assertEqual(2.0, res)

  def test_get_value_in_type_str__returns_str(self):
    res = get_value_in_type("abc", "2")
    self.assertTrue(isinstance(res, str))
    self.assertEqual("2", res)

  def test_get_value_in_type_bool__returns_bool(self):
    res = get_value_in_type(False, "true")
    self.assertTrue(isinstance(res, bool))
    self.assertEqual(True, res)

  def test_get_value_in_type_list__returns_list(self):
    res = get_value_in_type(["abc"], "2")
    self.assertTrue(isinstance(res, list))
    self.assertEqual(["2"], res)

  # NOTE: I replaced tf hparams with dataclasses
  # def hp_raw(hparams) -> Dict[str, Any]:
  #   return hparams.values()

  # def hp_from_raw(raw: Dict[str, Any]) -> tf.contrib.training.HParams:
  #   return tf.contrib.training.HParams(**raw)

  # def overwrite_custom_tf_hparams(hparams: Any, custom_hparams: Optional[Dict[str, str]]) -> None:
  #   # Note: This method does no type conversion from str.
  #   # hparams.override_from_dict(custom_hparams)
  #   # E.g.: ValueError: Could not cast hparam 'epochs' of type '<class 'int'>' from value '10'
  #   if custom_hparams is not None:
  #     for param_name, current_value in hparams.values().items():
  #       if param_name in custom_hparams.keys():
  #         new_value = get_value_in_type(current_value, custom_hparams[param_name])
  #         hparams.set_hparam(param_name, new_value)

  # def test_overwrite_custom_hparams(self):
  #   hparams = tf.contrib.training.HParams(
  #     epochs=500,
  #     seed=1234,
  #   )

  #   custom_hparams = {
  #     "epochs": "10",
  #     "learning_rate": "0.5"
  #   }

  #   overwrite_custom_tf_hparams(hparams, custom_hparams)

  #   self.assertEqual(10, hparams.epochs)
  #   self.assertTrue(isinstance(hparams.epochs, int))
  #   self.assertEqual(1234, hparams.seed)
  #   self.assertTrue(isinstance(hparams.seed, int))

  # def test_overwrite_custom_hparams_none__does_nothing(self):
  #   hparams = tf.contrib.training.HParams(
  #     epochs=500,
  #     seed=1234,
  #   )

  #   custom_hparams = None

  #   overwrite_custom_tf_hparams(hparams, custom_hparams)

  #   self.assertEqual(500, hparams.epochs)
  #   self.assertTrue(isinstance(hparams.epochs, int))
  #   self.assertEqual(1234, hparams.seed)
  #   self.assertTrue(isinstance(hparams.seed, int))

  def test_check_is_first__it0_is_false(self):
    res = check_is_first(iteration=0)
    self.assertFalse(res)

  def test_check_is_first__it1_is_true(self):
    res = check_is_first(iteration=1)
    self.assertTrue(res)

  def test_check_is_first__it2_is_false(self):
    res = check_is_first(iteration=2)
    self.assertFalse(res)

  def test_check_is_last__it0_is_false(self):
    res = check_is_last(iteration=0, epochs=2, batch_iterations=3)
    self.assertFalse(res)

  def test_check_is_last__it5_is_true(self):
    res = check_is_last(iteration=6, epochs=2, batch_iterations=3)
    self.assertTrue(res)

  def test_check_is_save_iteration__cu0_pc5_is_false(self):
    res = check_is_save_iteration(iteration=0, iters_per_checkpoint=5)
    self.assertFalse(res)

  def test_check_is_save_iteration__cu5_pc5_is_true(self):
    res = check_is_save_iteration(iteration=5, iters_per_checkpoint=5)
    self.assertTrue(res)

  def test_check_is_save_iteration__cu10_pc5_is_true(self):
    res = check_is_save_iteration(iteration=10, iters_per_checkpoint=5)
    self.assertTrue(res)

  def test_check_is_save_iteration__cu11_pc5_is_false(self):
    res = check_is_save_iteration(iteration=11, iters_per_checkpoint=5)
    self.assertFalse(res)

  def test_check_is_save_iteration__cu3_pc1_is_true(self):
    res = check_is_save_iteration(iteration=3, iters_per_checkpoint=1)
    self.assertTrue(res)

  def test_check_is_save_iteration__cu3_pc0_is_false(self):
    res = check_is_save_iteration(iteration=3, iters_per_checkpoint=0)
    self.assertFalse(res)

  def test_check_is_last_batch_iteration__it0_tot5_is_false(self):
    res = check_is_last_batch_iteration(iteration=0, batch_iterations=5)
    self.assertFalse(res)

  def test_check_is_last_batch_iteration__it1_tot5_is_false(self):
    res = check_is_last_batch_iteration(iteration=1, batch_iterations=5)
    self.assertFalse(res)

  def test_check_is_last_batch_iteration__it5_tot5_is_true(self):
    res = check_is_last_batch_iteration(iteration=5, batch_iterations=5)
    self.assertTrue(res)

  def test_check_is_save_epoch__cu5_pc0_is_false(self):
    res = check_is_save_epoch(
      epoch=5,
      epochs_per_checkpoint=0
    )

    self.assertFalse(res)

  def test_check_is_save_epoch__cu0_pc1_is_true(self):
    res = check_is_save_epoch(
      epoch=0,
      epochs_per_checkpoint=1,
    )

    self.assertTrue(res)

  def test_check_is_save_epoch__cu0_pc5_is_false(self):
    res = check_is_save_epoch(
      epoch=0,
      epochs_per_checkpoint=5,
    )

    self.assertTrue(res)

  def test_check_is_save_epoch__cu5_pc5_is_true(self):
    res = check_is_save_epoch(
      epoch=5,
      epochs_per_checkpoint=5,
    )

    self.assertTrue(res)

  # region iteration_to_epoch
  def test_iteration_to_epoch_it1_tot2_is_0(self):
    res = iteration_to_epoch(iteration=1, batch_iterations=2)
    self.assertEqual(0, res)

  def test_iteration_to_epoch_it2_tot2_is_0(self):
    res = iteration_to_epoch(iteration=2, batch_iterations=2)
    self.assertEqual(0, res)

  def test_iteration_to_epoch_it3_tot2_is_1(self):
    res = iteration_to_epoch(iteration=3, batch_iterations=2)
    self.assertEqual(1, res)

  def test_iteration_to_epoch_it4_tot2_is_1(self):
    res = iteration_to_epoch(iteration=4, batch_iterations=2)
    self.assertEqual(1, res)

  def test_iteration_to_epoch_it5_tot2_is_2(self):
    res = iteration_to_epoch(iteration=5, batch_iterations=2)
    self.assertEqual(2, res)

  def test_iteration_to_epoch_it6_tot2_is_2(self):
    res = iteration_to_epoch(iteration=5, batch_iterations=2)
    self.assertEqual(2, res)
  # endregion

  # region get_continue_epoch
  def test_get_continue_epoch_cur0_tot2_is_0(self):
    res = get_continue_epoch(current_iteration=0, batch_iterations=2)
    self.assertEqual(0, res)

  def test_get_continue_epoch_cur1_tot2_is_0(self):
    res = get_continue_epoch(current_iteration=1, batch_iterations=2)
    self.assertEqual(0, res)

  def test_get_continue_epoch_cur2_tot2_is_1(self):
    res = get_continue_epoch(current_iteration=2, batch_iterations=2)
    self.assertEqual(1, res)

  def test_get_continue_epoch_cur3_tot2_is_1(self):
    res = get_continue_epoch(current_iteration=3, batch_iterations=2)
    self.assertEqual(1, res)

  def test_get_continue_epoch_cur4_tot2_is_2(self):
    res = get_continue_epoch(current_iteration=4, batch_iterations=2)
    self.assertEqual(2, res)

  def test_get_continue_epoch_cur5_tot2_is_2(self):
    res = get_continue_epoch(current_iteration=5, batch_iterations=2)
    self.assertEqual(2, res)
  # endregion

  # region iteration_to_batch_iteration
  def test_iteration_to_batch_iteration_it1_tot2_is_0(self):
    res = iteration_to_batch_iteration(iteration=1, batch_iterations=2)
    self.assertEqual(0, res)

  def test_iteration_to_batch_iteration_it2_tot2_is_1(self):
    res = iteration_to_batch_iteration(iteration=2, batch_iterations=2)
    self.assertEqual(1, res)

  def test_iteration_to_batch_iteration_it3_tot2_is_0(self):
    res = iteration_to_batch_iteration(iteration=3, batch_iterations=2)
    self.assertEqual(0, res)

  def test_iteration_to_batch_iteration_it4_tot2_is_1(self):
    res = iteration_to_batch_iteration(iteration=4, batch_iterations=2)
    self.assertEqual(1, res)

  def test_iteration_to_batch_iteration_it5_tot2_is_0(self):
    res = iteration_to_batch_iteration(iteration=5, batch_iterations=2)
    self.assertEqual(0, res)
  # endregion

  # region get_continue_batch_iteration
  def test_get_continue_iteration_cur0_tot2_is_0(self):
    res = get_continue_batch_iteration(iteration=0, batch_iterations=2)
    self.assertEqual(0, res)

  def test_get_continue_iteration_cur1_tot2_is_1(self):
    res = get_continue_batch_iteration(iteration=1, batch_iterations=2)
    self.assertEqual(1, res)

  def test_get_continue_iteration_cur2_tot2_is_0(self):
    res = get_continue_batch_iteration(iteration=2, batch_iterations=2)
    self.assertEqual(0, res)

  def test_get_continue_iteration_cur3_tot2_is_1(self):
    res = get_continue_batch_iteration(iteration=3, batch_iterations=2)
    self.assertEqual(1, res)

  def test_get_continue_iteration_cur4_tot2_is_0(self):
    res = get_continue_batch_iteration(iteration=4, batch_iterations=2)
    self.assertEqual(0, res)

  def test_get_continue_iteration_cur5_tot2_is_1(self):
    res = get_continue_batch_iteration(iteration=5, batch_iterations=2)
    self.assertEqual(1, res)
  # endregion

  def test_dummy_dataset(self):
    dataset = DummyDataset([1, 2, 3])
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    self.assertEqual(3, len(data_loader))
    self.assertEqual([1, 2, 3], list(data_loader))

  def test_skip_batch_cur0_is_123(self):
    data_loader = DataLoader(DummyDataset([1, 2, 3]), batch_size=1, shuffle=False)
    continue_batch_iteration = 0
    result = []
    for batch_iteration, _ in enumerate(data_loader):
      result.append(skip_batch(continue_batch_iteration, batch_iteration))
    self.assertEqual([False, False, False], result)

  def test_skip_batch_cur1_is_23(self):
    data_loader = DataLoader(DummyDataset([1, 2, 3]), batch_size=1, shuffle=False)
    continue_batch_iteration = 1
    result = []
    for batch_iteration, _ in enumerate(data_loader):
      result.append(skip_batch(continue_batch_iteration, batch_iteration))
    self.assertEqual([True, False, False], result)

  def test_skip_batch_cur2_is_3(self):
    data_loader = DataLoader(DummyDataset([1, 2, 3]), batch_size=1, shuffle=False)
    continue_batch_iteration = 2
    result = []
    for batch_iteration, _ in enumerate(data_loader):
      result.append(skip_batch(continue_batch_iteration, batch_iteration))
    self.assertEqual([True, True, False], result)


if __name__ == '__main__':
  suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
  unittest.TextTestRunner(verbosity=2).run(suite)
