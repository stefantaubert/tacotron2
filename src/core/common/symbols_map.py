import logging
from collections import OrderedDict
from typing import List, Optional
from typing import OrderedDict as OrderedDictType
from typing import Set, Tuple

from src.core.common.language import Language
from src.core.common.symbol_id_dict import SymbolIdDict
from src.core.common.text import text_to_symbols
from src.core.common.utils import get_sorted_set, parse_json, save_json


class SymbolsMap(OrderedDict):

  @classmethod
  def from_two_sets(cls, map_from: set, map_to: set):
    #only_a = list(sorted(list(symbolsA)))
    in_both = list(sorted(list(map_from.intersection(map_to))))
    sym_mapping = cls([(symb, symb) for symb in in_both])

    symbs_in_map_to_without_mapping = map_to.difference(map_from)
    for symb in get_sorted_set(symbs_in_map_to_without_mapping):
      sym_mapping[symb] = ""

    return sym_mapping

  @classmethod
  def from_tuples(cls, map_to_from: List[Tuple[str, str]]):
    return cls(map_to_from)

  def save(self, file_path: str):
    save_json(file_path, self)

  @classmethod
  def load(cls, file_path: str):
    data = parse_json(file_path)
    return cls(data)

  def apply_to_symbols(self, symbols: List[str]):
    res = []
    for symbol in symbols:
      if symbol in self.keys():
        res.append(self[symbol])
      else:
        res.append(symbol)
    return res

  def get_symbols_without_mapping(self) -> Set[str]:
    result = [map_to for map_to, map_from in self.items() if map_from == ""]
    return result

  def filter(self, dest_symbols: set, orig_symbols: set, logger: logging.Logger):
    for map_to_symbol, map_from_symbol in self.items():
      if map_to_symbol not in dest_symbols:
        logger.info(
          f"Symbol '{map_to_symbol}' doesn't exist in destination symbol set. Ignoring mapping from '{map_from_symbol}'.")
        self.pop(map_to_symbol)
      elif map_from_symbol not in orig_symbols:
        logger.info(
          f"Symbol '{map_from_symbol}' doesn't exist in original symbol set. Ignoring mapping to '{map_to_symbol}'.")
        self.pop(map_to_symbol)
      else:
        #result[map_to_symbol] = map_from_symbol
        logger.info(
          f"Keeped mapping of symbol '{map_from_symbol}' to symbol '{map_to_symbol}'.")

  def try_fix_symbols_without_mapping(self, old_map: OrderedDict):
    """returns True, if a new key was added"""
    for to_symbol, from_symbol in self.items():
      if from_symbol == "" and to_symbol in old_map and old_map[to_symbol] != "":
        self[to_symbol] = old_map[to_symbol]

  def has_new_to_mappings(self, old_map: OrderedDict) -> bool:
    for new_key in self.keys():
      if new_key not in old_map.keys():
        return True
    return False


def sort_map_after_map_from_symbol(symb_map: SymbolsMap):
  new_map = SymbolsMap(
    sorted(symb_map.items(), key=lambda x: x[1], reverse=False))
  return new_map


def create_inference_map(model_symbols: set, corpora: str, lang: Language, ignore_tones: Optional[bool], ignore_arcs: Optional[bool], existing_map: Optional[SymbolsMap], logger: logging.Logger) -> Tuple[SymbolsMap, List[str]]:
  raw_dest_symbols = text_to_symbols(
    corpora,
    lang=lang,
    ignore_arcs=ignore_arcs,
    ignore_tones=ignore_tones
  )

  dest = set(raw_dest_symbols)
  return create_or_update_map(model_symbols, dest, existing_map, logger)


def create_or_update_map(orig: Set[str], dest: Set[str], existing_map: Optional[SymbolsMap], logger: logging.Logger) -> Tuple[SymbolsMap, List[str]]:
  dest_map = SymbolsMap.from_two_sets(orig, dest)
  if existing_map:
    # The usecase is, when thchs map without tones exist and I want to create a map for thchs with tones.
    dest_map.try_fix_symbols_without_mapping(existing_map)
    if not dest_map.has_new_to_mappings(existing_map):
      logger.info("There were no new symbols in the destination symbol set.")
  dest_map = sort_map_after_map_from_symbol(dest_map)
  return dest_map, get_sorted_set(orig)


def get_map(dest_symbols: Set[str], orig_symbols: Set[str], symbols_mapping: Optional[SymbolsMap], logger: logging.Logger) -> SymbolsMap:
  if symbols_mapping is None:
    symbols_mapping = SymbolsMap.from_two_sets(orig_symbols, dest_symbols)
  else:
    symbols_mapping.filter(dest_symbols, orig_symbols, logger)

  return symbols_mapping


def symbols_map_to_symbols_ids_map(dest_symbols: SymbolIdDict, orig_symbols: SymbolIdDict, symbols_mapping: SymbolsMap, logger: logging.Logger) -> OrderedDictType[int, int]:
  result: OrderedDictType[int, int] = OrderedDict()

  for map_to_symbol, map_from_symbol in symbols_mapping.items():
    assert dest_symbols.symbol_exists(map_to_symbol)
    assert orig_symbols.symbol_exists(map_from_symbol)

    map_from_symbol_id = orig_symbols.get_id(map_from_symbol)
    map_to_symbol_id = dest_symbols.get_id(map_to_symbol)
    result[map_to_symbol_id] = map_from_symbol_id
    logger.info(
      f"Mapped symbol '{map_from_symbol}' ({map_from_symbol_id}) to symbol '{map_to_symbol}' ({map_to_symbol_id})")

  return result
