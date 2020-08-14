import os

def get_last_checkpoint(checkpoint_dir) -> str:
  #checkpoint_dir = get_checkpoint_dir(training_dir_path)
  _, _, filenames = next(os.walk(checkpoint_dir))
  filenames = [x[:-3] for x in filenames if ".pt" in x]
  at_least_one_checkpoint_exists = len(filenames) > 0
  if at_least_one_checkpoint_exists:
    last_checkpoint = f"{max(list(map(int, filenames)))}.pt"
    return last_checkpoint
  return ""