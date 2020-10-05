from collections import OrderedDict
from logging import Logger
from typing import Optional
from typing import OrderedDict as OrderedDictType

from torch import Tensor

from src.core.common.symbol_id_dict import SymbolIdDict
from src.core.common.symbols_map import (SymbolsMap, get_map,
                                         symbols_map_to_symbols_ids_map)
from src.core.tacotron.hparams import HParams
from src.core.tacotron.model import get_symbol_weights
from src.core.tacotron.model_symbols import get_model_symbol_id


def symbols_ids_map_to_model_symbols_ids_map(symbols_id_map: OrderedDictType[int, int], n_accents: int, n_symbols: int, accents_use_own_symbols: bool) -> OrderedDictType[int, int]:
  res: OrderedDictType[int, int] = OrderedDict()

  for accent_id in range(n_accents):
    for map_to_symbol_id, map_from_symbol_id in symbols_id_map.items():

      map_to_model_id = get_model_symbol_id(
        map_to_symbol_id,
        accent_id,
        n_symbols,
        accents_use_own_symbols
      )

      res[map_to_model_id] = symbols_id_map[map_from_symbol_id]

    if not accents_use_own_symbols:
      break

  return res


def map_weights(model_symbols_id_map: OrderedDictType[int, int], model_weights, trained_weights, logger: Logger):
  for map_to_model_symbol_id, map_from_symbol_id in model_symbols_id_map.items():
    assert 0 <= map_to_model_symbol_id < model_weights.shape[0]
    assert 0 <= map_from_symbol_id < trained_weights.shape[0]

    logger.debug(f"Mapped {map_from_symbol_id} to {map_to_model_symbol_id}.")
    model_weights[map_to_model_symbol_id] = trained_weights[map_from_symbol_id]


def get_mapped_symbol_weights(model_symbols: SymbolIdDict, trained_weights: Tensor, trained_symbols: SymbolIdDict, custom_mapping: Optional[SymbolsMap], hparams: HParams, logger: Logger) -> Tensor:
  symbols_match_not_model = trained_weights.shape[0] != len(trained_symbols)
  if symbols_match_not_model:
    logger.exception(
      f"Weights mapping: symbol space from pretrained model ({trained_weights.shape[0]}) did not match amount of symbols ({len(trained_symbols)}).")
    raise Exception()

  symbols_map = get_map(
    dest_symbols=model_symbols.get_all_symbols(),
    orig_symbols=trained_symbols.get_all_symbols(),
    symbols_mapping=custom_mapping,
    logger=logger
  )

  symbols_id_map = symbols_map_to_symbols_ids_map(
    dest_symbols=model_symbols,
    orig_symbols=trained_symbols,
    symbols_mapping=symbols_map,
    logger=logger
  )

  model_symbols_id_map = symbols_ids_map_to_model_symbols_ids_map(
    symbols_id_map,
    hparams.n_accents,
    n_symbols=hparams.n_symbols,
    accents_use_own_symbols=hparams.accents_use_own_symbols
  )

  model_weights = get_symbol_weights(hparams)

  map_weights(
    model_symbols_id_map=model_symbols_id_map,
    model_weights=model_weights,
    trained_weights=trained_weights,
    logger=logger
  )

  symbols_wo_mapping = symbols_map.get_symbols_without_mapping()
  not_existing_symbols = model_symbols.get_all_symbols() - symbols_map.keys()
  no_mapping = symbols_wo_mapping | not_existing_symbols
  if len(no_mapping) > 0:
    logger.warning(f"Following symbols were not mapped: {no_mapping}")
  else:
    logger.info("All symbols were mapped.")

  return model_weights
