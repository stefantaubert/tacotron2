import random
from math import ceil
from typing import Dict, List, Set, Tuple, Type, TypeVar, Union

_T = TypeVar('_T')


def contains_all(text_data_diphones: List[Union[Set[_T], List[_T]]], which_should_contained: Set[_T]) -> Tuple[bool, Set[_T]]:
  rest = which_should_contained
  for which_should_contained in text_data_diphones:
    new_rest = rest.difference(set(which_should_contained))
    rest = new_rest
    if len(rest) == 0:
      return True, rest
  return False, rest


if __name__ == "__main__":
  res = contains_all(
    text_data_diphones=[set(["a", "b"])],
    which_should_contained=set(["a", "b", "c"])
  )

  res = contains_all(
    text_data_diphones=[set(["a", "b", "c"])],
    which_should_contained=set(["a", "b", "c"])
  )
