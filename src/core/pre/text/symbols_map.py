import logging
from collections import OrderedDict
import torch
from math import sqrt
from src.core.pre.text.symbol_converter import SymbolConverter
from src.core.pre.text.ipa2symb import extract_from_sentence
from typing import OrderedDict as OrderedDictType, Tuple, List, Set, Optional
from enum import IntEnum
from src.core.common import parse_json, save_json


class SymbolsMap(OrderedDictType[str, str]):
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

  def has_empty_mapping(self) -> bool:
    for v in self.values():
      if v == "":
        return True
    return False

  def apply_to_symbols(self, symbols: List[str]):
    res = []
    for s in symbols:
      if s in self.keys():
        res.append(self[s])
      else:
        res.append(s)
    return res
  
def sort_map_after_map_from_symbol(symb_map: SymbolsMap):
  new_map = SymbolsMap(sorted(symb_map.items(), key = lambda x: x[1], reverse=False))
  return new_map

def create_weights_map(orig_conv: SymbolConverter, dest_conv: SymbolConverter, existing_map: Optional[SymbolsMap] = None, logger: logging.Logger = logging.getLogger()) -> Tuple[SymbolsMap, List[str]]:
  orig = set(orig_conv.get_symbols())
  dest = set(dest_conv.get_symbols())
  weights_map = SymbolsMap.from_two_sets(orig, dest)
  if existing_map:
    # The usecase is, when thchs map without tones exist and I want to create a map for thchs with tones.
    new_keys = update_map(old_map=existing_map, new_map=weights_map)
    if not new_keys:
      logger.info("There were no new symbols in the destination symbol set.")
  weights_map = sort_map_after_map_from_symbol(weights_map)
  return weights_map, get_sorted_set(orig)

def update_map(old_map: SymbolsMap, new_map: SymbolsMap) -> bool:
  """returns True, if a new key was added"""
  for to_symbol, from_symbol in new_map.items():
    if from_symbol == "" and to_symbol in old_map and old_map[to_symbol] != "":
      new_map[to_symbol] = old_map[to_symbol]
  for new_key in new_map.keys():
    if new_key not in old_map.keys():
      return True
  return False

def create_inference_map(model_symb_conv: SymbolConverter, corpora: str, is_ipa: str = False, ignore_tones: Optional[bool] = None, ignore_arcs: Optional[bool] = None, existing_map: Optional[SymbolsMap] = None, logger: logging.Logger = logging.getLogger()) -> Tuple[SymbolsMap, List[str]]:
  model_symbs = set(model_symb_conv.get_symbols())
  return create_inference_map_core(model_symbs, corpora, is_ipa, ignore_tones, ignore_arcs, existing_map, logger)

def create_inference_map_core(model_symbols: set, corpora: str, is_ipa: str = False, ignore_tones: Optional[bool] = None, ignore_arcs: Optional[bool] = None, existing_map: Optional[SymbolsMap] = None, logger: logging.Logger = logging.getLogger()) -> Tuple[SymbolsMap, List[str]]:
  dest_symbols = set(extract_symbols(corpora, is_ipa, ignore_tones, ignore_arcs))
  infer_map = SymbolsMap.from_two_sets(model_symbols, dest_symbols)
  if existing_map:
    new_keys = update_map(old_map=existing_map, new_map=infer_map)
    if not new_keys:
      logger.info("There were no new symbols in the corpora.")
  infer_map = sort_map_after_map_from_symbol(infer_map)
  return infer_map, get_sorted_set(model_symbols)

def get_sorted_set(s: Set[str]) -> List[str]:
  res: List[str] = list(sorted(list(s)))
  return res

def extract_symbols(corpora: str, is_ipa: str = False, ignore_tones: Optional[bool] = None, ignore_arcs: Optional[bool] = None) -> List[str]:
  if is_ipa:
    raw_dest_symbols = extract_from_sentence(corpora, ignore_arcs=ignore_arcs, ignore_tones=ignore_tones, replace_unknown_ipa_by="_")
  else:
    raw_dest_symbols = list(corpora)
  return raw_dest_symbols

def create_symbols_map(dest_symbols: set, orig_symbols: set, symbols_mapping: Optional[SymbolsMap] = None, logger: logging.Logger = logging.getLogger()) -> SymbolsMap:
  result = SymbolsMap()
  if not symbols_mapping:
    symbols_mapping = SymbolsMap.from_two_sets(orig_symbols, dest_symbols)
    logger.info(f"intersecting symbols {orig_symbols.intersection(dest_symbols)}")
  not_mapped = set()
  for map_to_symbol, map_from_symbol in symbols_mapping.items():
    if map_to_symbol not in dest_symbols:
      logger.info(f"Symbol '{map_to_symbol}' doesn't exist in destination symbol set. Ignoring mapping from '{map_from_symbol}'.")
    elif not map_from_symbol:
      logger.info(f"Symbol '{map_to_symbol}' has no mapping assigned.")
      not_mapped.add(map_to_symbol)
    elif map_from_symbol not in orig_symbols:
      logger.info(f"Symbol '{map_from_symbol}' doesn't exist in original symbol set. Ignoring mapping to '{map_to_symbol}'.")
      not_mapped.add(map_to_symbol)
    else:
      result[map_to_symbol] = map_from_symbol
      logger.info(f"Mapped symbol '{map_from_symbol}' to symbol '{map_to_symbol}'")

  unmapped_symbols = dest_symbols.difference(set(symbols_mapping.keys())).union(not_mapped)
  if len(unmapped_symbols):
    logger.info(f"Symbols without initialized mapping: {unmapped_symbols}")
  else:
    logger.info("All symbols were mapped.")

  return result

def get_symbols_id_mapping(dest_symbols: SymbolConverter, orig_symbols: SymbolConverter, symbols_mapping: Optional[SymbolsMap] = None, logger: logging.Logger = logging.getLogger()) -> OrderedDictType[int, int]:
  map_from = set(orig_symbols.get_symbols())
  map_to = set(dest_symbols.get_symbols())
  symbols_map = create_symbols_map(map_to, map_from, symbols_mapping, logger)

  result: OrderedDictType[int, int] = OrderedDict()

  for map_to_symbol, map_from_symbol in symbols_map.items():
    assert dest_symbols.symbol_exists(map_to_symbol)
    assert orig_symbols.symbol_exists(map_from_symbol)
    map_from_symbol_id = orig_symbols.symbol_to_id(map_from_symbol, subset_id_if_multiple=0)
    map_to_symbol_id = dest_symbols.symbol_to_id(map_to_symbol, subset_id_if_multiple=1)
    result[map_to_symbol_id] = map_from_symbol_id
    logger.info(f"Mapped symbol '{map_from_symbol}' ({map_from_symbol_id}) to symbol '{map_to_symbol}' ({map_to_symbol_id})")

  return result

