from src.core.common import get_subdir

def get_pre_dir(base_dir: str, create: bool = False):
  return get_subdir(base_dir, 'pre', create)
