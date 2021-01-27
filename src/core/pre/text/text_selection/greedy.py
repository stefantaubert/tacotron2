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


def set_cover_dict_max_length(subsets: Dict[int, Set[_T]], lengths: Dict[int, int], max_length: int) -> Tuple[bool, Dict[int, Set[_T]], int]:
  """Find a family of subsets that covers the universal set"""
  elements: Set[_T] = set(e for s in subsets.values() for e in s)
  covered: Set[_T] = set()
  cover: Dict[int, Set[_T]] = {}
  current_length = 0
  finished = True
  while covered != elements:
    k, v_subset = max(subsets.items(), key=lambda x: get_new_units_count(x[1], covered))
    length = lengths[k]
    new_length = length + current_length
    if new_length <= max_length:
      current_length = new_length
      cover[k] = v_subset
      covered |= v_subset
    else:
      finished = False
      break

  return finished, cover, current_length


def get_new_units_count(subset: Set[_T], already_covered: Set[_T]) -> int:
  difference = subset - already_covered
  res = len(difference)
  return res


def set_cover_n(subsets: Dict[int, Set[_T]], n: int) -> Dict[int, Set[_T]]:
  result = {}
  rest = {k: v for k, v in subsets.items()}
  counter = 0
  while len(rest) > 0 and counter < n:
    tmp = set_cover_dict(rest)
    result.update(tmp)
    rest = {k: v for k, v in subsets.items() if k not in result}
    counter += 1
  return result


def set_cover_n_chars(subsets: Dict[int, Set[_T]], lengths: Dict[int, int], n_max: int) -> Tuple[Dict[int, Set[_T]], Dict[int, Set[_T]]]:
  result = {}
  rest = {k: v for k, v in subsets.items()}
  rest_lenghts = {k: v for k, v in lengths.items()}
  rest_char_count = n_max
  iterations = 0
  while len(rest) > 0:
    print(f"applying set cover algo for max {rest_char_count} chars...")
    f, tmp, c = set_cover_dict_max_length(rest, rest_lenghts, rest_char_count)
    print("done.")
    if c == 0:
      break
    iterations += 1
    result.update(tmp)
    old_rest_count = len(rest)
    rest = {k: v for k, v in subsets.items() if k not in result}
    rest_lenghts = {k: v for k, v in lengths.items() if k not in result}
    print(f"Reduced pool from {old_rest_count} to {len(rest)}")
    rest_char_count -= c
  final_count = n_max - rest_char_count
  print(
    f"Extracted {final_count} chars from {len(result)} out of {len(subsets)} utterances ({len(rest)} remain) in {iterations} iterations.")
  return result, rest, final_count
