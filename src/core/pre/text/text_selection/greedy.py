from typing import Dict, List, Set, Tuple, Type, TypeVar, Union

_T = TypeVar('_T')


def set_cover(subsets: List[Set[_T]]) -> List[Set[_T]]:
  """Find a family of subsets that covers the universal set"""
  elements: Set[_T] = set(e for s in subsets for e in s)
  # Check the subsets cover the universe
  covered: Set[_T] = set()
  cover: List[Set[_T]] = []
  # Greedily add the subsets with the most uncovered points
  while covered != elements:
    subset = max(subsets, key=lambda subset: get_new_units_count(subset, covered))
    cover.append(subset)
    covered |= subset

  return cover


def set_cover_dict(subsets: Dict[int, Set[_T]]) -> Dict[int, Set[_T]]:
  """Find a family of subsets that covers the universal set"""
  elements: Set[_T] = set(e for s in subsets.values() for e in s)
  covered: Set[_T] = set()
  cover: Dict[int, Set[_T]] = {}

  while covered != elements:
    k, v_subset = max(subsets.items(), key=lambda x: get_new_units_count(x[1], covered))
    cover[k] = v_subset
    covered |= v_subset

  return cover


def get_new_units_count(subset: Set[_T], already_covered: Set[_T]) -> int:
  difference = subset - already_covered
  res = len(difference)
  return res


def set_cover_n(subsets: Dict[int, Set[_T]], n: int) -> Dict[int, Set[_T]]:
  result = {}
  rest = {k: v for k, v in subsets.items()}
  counter = 0
  while len(rest) > 0 and n < counter:
    tmp = set_cover_dict(rest)
    result.update(tmp)
    rest = {k: v for k, v in subsets.items() if k not in result}
    counter += 1
  return result
