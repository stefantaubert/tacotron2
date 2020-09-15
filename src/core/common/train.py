import os
from dataclasses import dataclass
from math import floor
from typing import List, Optional, Tuple

PYTORCH_EXT = ".pt"


def get_pytorch_filename(name: str) -> str:
  return f"{name}{PYTORCH_EXT}"


def get_pytorch_basename(filename: str):
  return filename[:-len(PYTORCH_EXT)]


def is_pytorch_file(filename: str):
  return filename.endswith(PYTORCH_EXT)


def get_last_checkpoint(checkpoint_dir: str) -> Tuple[str, int]:
  #checkpoint_dir = get_checkpoint_dir(training_dir_path)
  its = get_all_checkpoint_iterations(checkpoint_dir)
  at_least_one_checkpoint_exists = len(its) > 0
  if not at_least_one_checkpoint_exists:
    raise Exception("No checkpoint iteration found!")
  last_iteration = max(its)
  last_checkpoint = get_pytorch_filename(last_iteration)
  checkpoint_path = os.path.join(checkpoint_dir, last_checkpoint)
  return checkpoint_path, last_iteration


def get_all_checkpoint_iterations(checkpoint_dir: str) -> List[int]:
  _, _, filenames = next(os.walk(checkpoint_dir))
  checkpoints_str = [get_pytorch_basename(x)
                     for x in filenames if is_pytorch_file(x)]
  checkpoints = list(sorted(list(map(int, checkpoints_str))))
  return checkpoints


def get_custom_checkpoint(checkpoint_dir: str, custom_iteration: int) -> Tuple[str, int]:
  checkpoint_path = os.path.join(
    checkpoint_dir, get_pytorch_filename(custom_iteration))
  if not os.path.isfile(checkpoint_path):
    raise Exception(f"Checkpoint with iteration {custom_iteration} not found!")
  return checkpoint_path, custom_iteration


def get_custom_or_last_checkpoint(checkpoint_dir: str, custom_iteration: int) -> Tuple[str, int]:
  return get_custom_checkpoint(checkpoint_dir, custom_iteration) if custom_iteration else get_last_checkpoint(checkpoint_dir)


def get_formatted_current_total(current: int, total: int) -> str:
  return f"{str(current).zfill(len(str(total)))}/{total}"


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
  return save_epochs and epoch % epochs_per_checkpoint == 0


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
