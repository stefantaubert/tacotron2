import dataclasses
import os
from dataclasses import asdict, dataclass, replace
from logging import Logger
from math import floor, sqrt
from typing import Any, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union

import torch
from torch import Tensor, nn
from torch.optim.optimizer import \
    Optimizer  # pylint: disable=no-name-in-module
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.core.common.utils import get_filenames

PYTORCH_EXT = ".pt"

_T = TypeVar("_T")


def init_torch_seed(seed: int):
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)


def init_cuddn(enabled: bool):
  torch.backends.cudnn.enabled = enabled


def init_cuddn_benchmark(enabled: bool):
  torch.backends.cudnn.benchmark = enabled


def get_pytorch_filename(name: Union[str, int]) -> str:
  return f"{name}{PYTORCH_EXT}"


def get_pytorch_basename(filename: str):
  return filename[:-len(PYTORCH_EXT)]


def is_pytorch_file(filename: str):
  return filename.endswith(PYTORCH_EXT)


def get_last_checkpoint(checkpoint_dir: str) -> Tuple[str, int]:
  # checkpoint_dir = get_checkpoint_dir(training_dir_path)
  its = get_all_checkpoint_iterations(checkpoint_dir)
  at_least_one_checkpoint_exists = len(its) > 0
  if not at_least_one_checkpoint_exists:
    raise Exception("No checkpoint iteration found!")
  last_iteration = max(its)
  last_checkpoint = get_pytorch_filename(last_iteration)
  checkpoint_path = os.path.join(checkpoint_dir, last_checkpoint)
  return checkpoint_path, last_iteration


def get_all_checkpoint_iterations(checkpoint_dir: str) -> List[int]:
  filenames = get_filenames(checkpoint_dir)
  checkpoints_str = [get_pytorch_basename(x)
                     for x in filenames if is_pytorch_file(x)]
  checkpoints = list(sorted(list(map(int, checkpoints_str))))
  return checkpoints


def get_checkpoint(checkpoint_dir: str, iteration: int) -> Tuple[str, int]:
  checkpoint_path = os.path.join(
    checkpoint_dir, get_pytorch_filename(iteration))
  if not os.path.isfile(checkpoint_path):
    raise Exception(f"Checkpoint with iteration {iteration} not found!")
  return checkpoint_path, iteration


def get_custom_or_last_checkpoint(checkpoint_dir: str, custom_iteration: Optional[int]) -> Tuple[str, int]:
  return get_checkpoint(checkpoint_dir, custom_iteration) if custom_iteration is not None else get_last_checkpoint(checkpoint_dir)


def get_value_in_type(old_value: _T, new_value: str) -> _T:
  old_type = type(old_value)
  new_value_with_original_type = old_type(new_value)
  return new_value_with_original_type


def get_only_known_params(params: Dict[str, str], hparams: _T) -> Dict[str, str]:
  available_params = asdict(hparams)
  res = {k: v for k, v in params.items() if k in available_params.keys()}
  return res


def get_dataclass_from_dict(params: Dict[str, str], dc: Type[_T]) -> Tuple[_T, Set[str]]:
  field_names = {x.name for x in dataclasses.fields(dc)}
  res = {k: v for k, v in params.items() if k in field_names}
  ignored = {k for k in params.keys() if k not in field_names}
  return dc(**res), ignored


def check_has_unknown_params(params: Dict[str, str], hparams: _T) -> bool:
  available_params = asdict(hparams)
  for custom_hparam in params.keys():
    if custom_hparam not in available_params.keys():
      return True
  return False


def set_types_according_to_dataclass(params: Dict[str, str], hparams: _T) -> None:
  available_params = asdict(hparams)
  for custom_hparam, new_value in params.items():
    assert custom_hparam in available_params.keys()
    hparam_value = available_params[custom_hparam]
    params[custom_hparam] = get_value_in_type(hparam_value, new_value)


def update_learning_rate_optimizer(optimizer: Optimizer, learning_rate: float):
  for param_group in optimizer.param_groups:
    param_group['lr'] = learning_rate

def overwrite_custom_hparams(hparams_dc: _T, custom_hparams: Optional[Dict[str, str]]) -> _T:
  if custom_hparams is None:
    return hparams_dc

  # custom_hparams = get_only_known_params(custom_hparams, hparams_dc)
  if check_has_unknown_params(custom_hparams, hparams_dc):
    raise Exception()

  set_types_according_to_dataclass(custom_hparams, hparams_dc)

  result = replace(hparams_dc, **custom_hparams)
  return result


def get_uniform_weights(dimension: int, emb_dim: int) -> Tensor:
  # TODO check cuda is correct here
  weight = torch.zeros(size=(dimension, emb_dim), device="cuda")
  std = sqrt(2.0 / (dimension + emb_dim))
  val = sqrt(3.0) * std  # uniform bounds for std
  nn.init.uniform_(weight, -val, val)
  return weight


def update_weights(emb: nn.Embedding, weights: Tensor) -> None:
  emb.weight = nn.Parameter(weights)


def weights_to_embedding(weights: Tensor) -> nn.Embedding:
  embedding = nn.Embedding(weights.shape[0], weights.shape[1])
  update_weights(embedding, weights)
  return embedding


def copy_state_dict(state_dict: Dict[str, Tensor], to_model: nn.Module, ignore: List[str]):
  model_dict = {k: v for k, v in state_dict.items() if k not in ignore}
  update_state_dict(to_model, model_dict)


def update_state_dict(model: nn.Module, updates: Dict[str, Tensor]):
  dummy_dict = model.state_dict()
  dummy_dict.update(updates)
  model.load_state_dict(dummy_dict)


def log_hparams(hparams: _T, logger: Logger):
  logger.info("=== HParams ===")
  for param, val in asdict(hparams).items():
    logger.info(f"- {param} = {val}")
  logger.info("===============")



def get_formatted_current_total(current: int, total: int) -> str:
  return f"{str(current).zfill(len(str(total)))}/{total}"


def validate_model(model: nn.Module, criterion: nn.Module, val_loader: DataLoader, batch_parse_method) -> Tuple[float, Tuple[float, nn.Module, Tuple, Tuple]]:
  model.eval()
  res = []
  with torch.no_grad():
    total_val_loss = 0.0
    for batch in tqdm(val_loader):
      x, y = batch_parse_method(batch)
      y_pred = model(x)
      loss = criterion(y_pred, y)
      # if distributed_run:
      #   reduced_val_loss = reduce_tensor(loss.data, n_gpus).item()
      # else:
      #  reduced_val_loss = loss.item()
      reduced_val_loss = loss.item()
      res.append((reduced_val_loss, model, y, y_pred))
      total_val_loss += reduced_val_loss
    avg_val_loss = total_val_loss / len(val_loader)
  model.train()

  return avg_val_loss, res


@dataclass
class SaveIterationSettings():
  epochs: int
  batch_iterations: int
  save_first_iteration: bool
  save_last_iteration: bool
  iters_per_checkpoint: int
  epochs_per_checkpoint: int


def check_save_it(epoch: int, iteration: int, settings: SaveIterationSettings) -> bool:
  if check_is_first(iteration) and settings.save_first_iteration:
    return True

  if check_is_last(iteration, settings.epochs, settings.batch_iterations) and settings.save_last_iteration:
    return True

  if check_is_save_iteration(iteration, settings.iters_per_checkpoint):
    return True

  is_last_batch_iteration = check_is_last_batch_iteration(iteration, settings.batch_iterations)
  if is_last_batch_iteration and check_is_save_epoch(epoch, settings.epochs_per_checkpoint):
    return True

  return False


def get_next_save_it(iteration: int, settings: SaveIterationSettings) -> Optional[int]:
  result = iteration
  while result <= settings.epochs * settings.batch_iterations:
    epoch = iteration_to_epoch(result, settings.batch_iterations)
    if check_save_it(epoch, result, settings):
      return result
    result += 1
  return None


def check_is_first(iteration: int) -> bool:
  assert iteration >= 0
  # iteration=0 means no training was done yet
  return iteration == 1


def check_is_last(iteration: int, epochs: int, batch_iterations: int) -> bool:
  assert iteration >= 0
  return iteration == epochs * batch_iterations


def check_is_save_iteration(iteration: int, iters_per_checkpoint: int) -> bool:
  assert iteration >= 0
  save_iterations = iters_per_checkpoint > 0
  return iteration > 0 and save_iterations and iteration % iters_per_checkpoint == 0


def check_is_save_epoch(epoch: int, epochs_per_checkpoint: int) -> bool:
  assert epoch >= 0

  save_epochs = epochs_per_checkpoint > 0
  return save_epochs and ((epoch + 1) % epochs_per_checkpoint == 0)


def check_is_last_batch_iteration(iteration: int, batch_iterations: int):
  assert iteration >= 0
  assert batch_iterations > 0
  if iteration == 0:
    return False
  batch_iteration = iteration_to_batch_iteration(iteration, batch_iterations)
  is_last_batch_iteration = batch_iteration + 1 == batch_iterations
  return is_last_batch_iteration


def get_continue_epoch(current_iteration: int, batch_iterations: int) -> int:
  return iteration_to_epoch(current_iteration + 1, batch_iterations)


def skip_batch(continue_batch_iteration: int, batch_iteration: int):
  result = batch_iteration < continue_batch_iteration
  return result


def iteration_to_epoch(iteration: int, batch_iterations: int) -> int:
  """result: [0, inf)"""
  # Iteration 0 has no epoch.
  assert iteration > 0

  iteration_zero_based = iteration - 1
  epoch = floor(iteration_zero_based / batch_iterations)
  return epoch


def iteration_to_batch_iteration(iteration: int, batch_iterations: int) -> int:
  """result: [0, iterations)"""
  # Iteration 0 has no batch iteration.
  assert iteration > 0

  iteration_zero_based = iteration - 1
  batch_iteration = iteration_zero_based % batch_iterations
  return batch_iteration


def get_continue_batch_iteration(iteration: int, batch_iterations: int) -> int:
  return iteration_to_batch_iteration(iteration + 1, batch_iterations)


def filter_checkpoints(iterations: List[int], select: Optional[int], min_it: Optional[int], max_it: Optional[int]) -> List[int]:
  if select is None:
    select = 0
  if min_it is None:
    min_it = 0
  if max_it is None:
    max_it = max(iterations)
  process_checkpoints = [checkpoint for checkpoint in iterations if checkpoint %
                         select == 0 and min_it <= checkpoint <= max_it]

  return process_checkpoints
