import os
from argparse import ArgumentParser
from typing import Dict, List, Optional, Tuple


def parse_tuple_list(tuple_list: Optional[str] = None) -> Optional[List[Tuple]]:
  """ tuple_list: "a,b;c,d;... """
  if tuple_list is None:
    return None

  step1: List[str] = tuple_list.split(';')
  result: List[Tuple] = [tuple(x.split(',')) for x in step1]
  result = list(sorted(set(result)))
  return result


def add_base_dir(parser: ArgumentParser):
  assert "base_dir" in os.environ.keys()
  base_dir = os.environ["base_dir"]
  parser.set_defaults(base_dir=base_dir)


def split_hparams_string(hparams: Optional[str]) -> Optional[Dict[str, str]]:
  if hparams is None:
    return None

  assignments = hparams.split(",")
  result = dict([x.split("=") for x in assignments])
  return result

