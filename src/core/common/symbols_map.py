from collections import OrderedDict
from typing import List, Optional
from typing import OrderedDict as OrderedDictType
from typing import Set, Tuple

from src.core.common.symbol_id_dict import SymbolIdDict
from src.core.common.utils import (get_sorted_list_from_set, parse_json,
                                   save_json)


class SymbolsMap(OrderedDict):

  @classmethod
  def from_intersection(cls, map_from: set, map_to: set):
    #only_a = list(sorted(list(symbolsA)))
    in_both = list(sorted(list(map_from.intersection(map_to))))
    sym_mapping = cls([(symb, symb) for symb in in_both])

    symbs_in_map_to_without_mapping = map_to.difference(map_from)
    for symb in get_sorted_list_from_set(symbs_in_map_to_without_mapping):
      sym_mapping[symb] = ""

    return sym_mapping

  def save(self, file_path: str):
    save_json(file_path, self)

  @classmethod
  def load(cls, file_path: str):
    data = parse_json(file_path)
    return cls(data)

  def apply_to_symbols(self, symbols: List[str]) -> List[str]:
    res = []
    for symbol in symbols:
      if symbol in self.keys():
        res.append(self[symbol])
      else:
        res.append(symbol)
    return res

  def get_symbols_with_empty_mapping(self) -> Set[str]:
    result = {map_to for map_to, map_from in self.items() if map_from == ""}
    return result

  def pop_batch(self, batch: Set[str]) -> None:
    for k in batch:
      self.pop(k)

  def remove_entries_with_unknown_to_symbols(self, known_symbols: Set[str]) -> Set[str]:
    all_to_symbols = set(self.keys())
    remove_keys = all_to_symbols.difference(known_symbols)

    self.pop_batch(remove_keys)

    return remove_keys

  def get_unknown_from_symbols(self, known_symbols: Set[str]) -> Set[str]:
    result = {x for x, from_symbol in self.items() if from_symbol not in known_symbols}
    return result

  def set_unknown_from_symbols_empty(self, known_symbols: Set[str]) -> Set[str]:
    result = self.get_unknown_from_symbols(known_symbols)

    for k in result:
      self[k] = ""

    return result

  def remove_entries_with_unknown_from_symbols(self, known_symbols: Set[str]) -> Set[str]:
    result = self.get_unknown_from_symbols(known_symbols)

    self.pop_batch(result)

    return result

  def remove_unknown_symbols(self, known_from_symbols: Set[str], known_to_symbol: Set[str]) -> Set[str]:
    res = self.remove_entries_with_unknown_from_symbols(known_from_symbols)
    res |= self.remove_entries_with_unknown_to_symbols(known_to_symbol)
    return res

  def convert_to_symbols_ids_map(self, to_symbols: SymbolIdDict, from_symbols: SymbolIdDict) -> OrderedDictType[int, int]:
    result: OrderedDictType[int, int] = OrderedDict()

    for map_to_symbol, map_from_symbol in self.items():
      assert to_symbols.symbol_exists(map_to_symbol)
      assert from_symbols.symbol_exists(map_from_symbol)

      map_from_symbol_id = from_symbols.get_id(map_from_symbol)
      map_to_symbol_id = to_symbols.get_id(map_to_symbol)
      result[map_to_symbol_id] = map_from_symbol_id
      # logger.info(
      #  f"Mapped symbol '{map_from_symbol}' ({map_from_symbol_id}) to symbol '{map_to_symbol}' ({map_to_symbol_id})")

    return result

  # def filter(self, dest_symbols: set, orig_symbols: set, logger: logging.Logger):
  #   remove_keys = []
  #   for map_to_symbol, map_from_symbol in self.items():
  #     if map_to_symbol not in dest_symbols:
  #       logger.info(
  #         f"Symbol '{map_to_symbol}' doesn't exist in destination symbol set. Ignoring mapping from '{map_from_symbol}'.")
  #       remove_keys.append(map_to_symbol)
  #     elif map_from_symbol not in orig_symbols:
  #       logger.info(
  #         f"Symbol '{map_from_symbol}' doesn't exist in original symbol set. Ignoring mapping to '{map_to_symbol}'.")
  #       remove_keys.append(map_to_symbol)
  #     else:
  #       #result[map_to_symbol] = map_from_symbol
  #       logger.info(
  #         f"Keeped mapping of symbol '{map_from_symbol}' to symbol '{map_to_symbol}'.")

  #   for k in remove_keys:
  #     self.pop(k)

  def update_existing_to_mappings(self, new_map: OrderedDict):
    for to_symbol, from_symbol in new_map.items():
      if to_symbol in self.keys() and from_symbol != "":
        self[to_symbol] = from_symbol

  def add_only_new_mappings(self, new_map: OrderedDict):
    for to_symbol, from_symbol in new_map.items():
      if to_symbol not in self.keys():
        self[to_symbol] = from_symbol

  # def try_fix_symbols_without_mapping(self, old_map: OrderedDict):
  #   """returns True, if a new key was added"""
  #   for to_symbol, from_symbol in self.items():
  #     if from_symbol == "" and to_symbol in old_map and old_map[to_symbol] != "":
  #       self[to_symbol] = old_map[to_symbol]

  # def has_new_to_mappings(self, old_map: OrderedDict) -> bool:
  #   for new_key in self.keys():
  #     if new_key not in old_map.keys():
  #       return True
  #   return False


def sort_map_after_map_from_symbol(symb_map: SymbolsMap) -> SymbolsMap:
  new_map = SymbolsMap(
    sorted(symb_map.items(), key=lambda x: x[1], reverse=False))
  return new_map


def create_or_update_inference_map(orig: Set[str], dest: Set[str], existing_map: Optional[SymbolsMap], template_map: Optional[SymbolsMap]) -> Tuple[SymbolsMap, List[str]]:
  """keeps all dest symbols in existing map, even if thye were not in the dest set."""
  new_map = SymbolsMap.from_intersection(orig, dest)

  if existing_map is not None:
    dest_map = existing_map
    dest_map.add_only_new_mappings(new_map)
  else:
    dest_map = new_map

  return _apply_template_map(template_map, dest_map, orig)


def create_or_update_weights_map(orig: Set[str], dest: Set[str], existing_map: Optional[SymbolsMap], template_map: Optional[SymbolsMap]) -> Tuple[SymbolsMap, List[str]]:
  """removes all dest symbols in existing map, that were not in the dest set."""
  dest_map = SymbolsMap.from_intersection(orig, dest)

  if existing_map is not None:
    dest_map.update_existing_to_mappings(existing_map)

  return _apply_template_map(template_map, dest_map, orig)


def _apply_template_map(template_map, dest_map, orig):
  if template_map is not None:
    # The usecase is, when thchs map without tones exist and I want to create a map for thchs with tones.
    dest_map.update_existing_to_mappings(template_map)

  # This required for both inference and weight maps to manually set the unmapped
  dest_map.set_unknown_from_symbols_empty(orig)

  # if not dest_map.has_new_to_mappings(existing_map):
  #   logger.info("There were no new symbols in the destination symbol set.")
  dest_map = sort_map_after_map_from_symbol(dest_map)
  return dest_map, get_sorted_list_from_set(orig)
