from typing import List, Tuple, Optional

def parse_tuple_list(tuple_list: Optional[str] = None) -> Optional[List[Tuple]]:
  """ tuple_list: "a,b;c,d;... """
  if tuple_list != None:
    step1: List[str] = tuple_list.split(';')
    result: List[Tuple] = [tuple(x.split(',')) for x in step1]
    result = list(sorted(set(result)))
    return result
  else:
    return None
