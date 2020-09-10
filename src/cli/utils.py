import os
from argparse import ArgumentParser
from typing import List, Optional, Tuple


def parse_tuple_list(tuple_list: Optional[str] = None) -> Optional[List[Tuple]]:
  """ tuple_list: "a,b;c,d;... """
  if tuple_list is not None:
    step1: List[str] = tuple_list.split(';')
    result: List[Tuple] = [tuple(x.split(',')) for x in step1]
    result = list(sorted(set(result)))
    return result
  return None


def add_base_dir(parser: ArgumentParser):
  assert "base_dir" in os.environ.keys()
  base_dir = os.environ["base_dir"]
  parser.set_defaults(base_dir=base_dir)
