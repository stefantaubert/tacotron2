import logging
from collections import OrderedDict
import torch
from math import sqrt
from src.core.pre import SymbolConverter
from typing import OrderedDict as OrderedDictType
from enum import IntEnum
from src.core.common import parse_json, save_json

class SymbolsMap(OrderedDictType[str, str]):
  @classmethod
  def from_two_sets(cls, map_from: set, map_to: set):
    #only_a = list(sorted(list(symbolsA)))
    in_a_and_b = list(sorted(list(map_from.intersection(map_to))))
    sym_mapping = cls([(a, a) for a in in_a_and_b])

    only_in_b = list(sorted(list(map_to.difference(map_from))))
    for b in only_in_b:
      sym_mapping[b] = ""

    return sym_mapping

  def save(self, file_path: str):
    save_json(file_path, self)
  
  @classmethod
  def load(cls, file_path: str):
    data = parse_json(file_path)
    return cls(data)

def create_map_for(conv_map_from: SymbolConverter, conv_map_to: SymbolConverter) -> SymbolsMap:
  symbols_from = set(conv_map_from.get_symbols())
  symbols_to = set(conv_map_to.get_symbols())
  return SymbolsMap.from_two_sets(symbols_from, symbols_to)

def get_mapped_embedding_weights(model_weights: torch.Tensor, model_symbols: SymbolConverter, trained_weights: torch.Tensor, trained_symbols: SymbolConverter, symbols_map: SymbolsMap = None, logger: logging.Logger = logging.getLogger()) -> torch.Tensor:
  assert model_weights.shape[0] == model_symbols.get_symbol_ids_count()

  symbols_match_not_model = trained_weights.shape[0] != trained_symbols.get_symbol_ids_count()
  if symbols_match_not_model:
    logger.exception(f"Weights mapping: symbol space from pretrained model ({trained_weights.shape[0]}) did not match amount of symbols ({trained_symbols.get_symbol_ids_count()}).")
    raise Exception()

  if not symbols_map:
    map_from = set(trained_symbols.get_symbols())
    map_to = set(model_symbols.get_symbols())
    symbols_map = SymbolsMap.from_two_sets(map_from, map_to)
    logger.info(f"intersecting symbols {map_from.intersection(map_to)}")

  not_mapped = set()
  for map_to_symbol, map_from_symbol in symbols_map.items():
    if not model_symbols.symbol_exists(map_to_symbol):
      logger.info(f"Symbol '{map_to_symbol}' doesn't exist in destination symbol set. Ignoring mapping from '{map_from_symbol}'.")
    elif not map_from_symbol:
      logger.info(f"Symbol '{map_to_symbol}' has no mapping assigned.")
      not_mapped.add(map_to_symbol)
    elif not trained_symbols.symbol_exists(map_from_symbol):
      logger.info(f"Symbol '{map_from_symbol}' doesn't exist in pretrained model. Ignoring mapping to '{map_to_symbol}'.")
      not_mapped.add(map_to_symbol)
    else:
      map_from_symbol_id = trained_symbols.symbol_to_id(map_from_symbol, subset_id_if_multiple=0)
      map_to_symbol_id = model_symbols.symbol_to_id(map_to_symbol, subset_id_if_multiple=1)
      model_weights[map_to_symbol_id] = trained_weights[map_from_symbol_id]
      logger.info(f"Mapped pretrained weights from symbol '{map_from_symbol}' ({map_from_symbol_id}) to symbol '{map_to_symbol}' ({map_to_symbol_id})")

  unmapped_symbols = set(model_symbols.get_symbols()).difference(set(symbols_map.keys())).union(not_mapped)
  if len(unmapped_symbols):
    logger.info(f"Symbols without initialized mapping: {unmapped_symbols}")
  else:
    logger.info("All symbols were mapped.")

  return model_weights
