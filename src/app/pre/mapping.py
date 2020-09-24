import os
from typing import List, Optional, Set

from src.app.pre.io import get_text_dir, load_text_symbol_converter
from src.app.pre.prepare import get_prepared_dir, load_prep_symbol_converter
from src.app.utils import add_console_out_to_logger, init_logger
from src.core.common.symbols_map import SymbolsMap, create_or_update_map
from src.core.common.utils import get_subfolder_names

INFER_MAP_FN = "inference_map.json"
INFER_MAP_SYMB_FN = "inference_map.symbols"


def get_all_symbols(prep_dir: str) -> Set[str]:
  all_text_names = get_subfolder_names(prep_dir)
  all_symbols: Set[str] = set()
  for text_name in all_text_names:
    text_dir = get_text_dir(prep_dir, text_name, create=False)
    text_symbol_ids = load_text_symbol_converter(text_dir)
    all_symbols |= text_symbol_ids.get_all_symbols()

  return all_symbols


def save_infer_map(prep_dir: str, infer_map: SymbolsMap):
  infer_map.save(get_infer_map_path(prep_dir))


def get_infer_map_path(prep_dir: str) -> str:
  path = os.path.join(prep_dir, INFER_MAP_FN)
  return path


def load_infer_map(prep_dir: str) -> SymbolsMap:
  return SymbolsMap.load(get_infer_map_path(prep_dir))


def infer_map_exists(prep_dir: str) -> bool:
  path = os.path.join(prep_dir, INFER_MAP_FN)
  return os.path.isfile(path)


def save_weights_map(prep_dir: str, orig_prep_name: str, weights_map: SymbolsMap):
  path = os.path.join(prep_dir, f"{orig_prep_name}.json")
  weights_map.save(path)


def load_weights_map(prep_dir: str, orig_prep_name: str) -> SymbolsMap:
  path = os.path.join(prep_dir, f"{orig_prep_name}.json")
  return SymbolsMap.load(path)


def weights_map_exists(prep_dir: str, orig_prep_name: str) -> bool:
  path = os.path.join(prep_dir, f"{orig_prep_name}.json")
  return os.path.isfile(path)


def try_load_symbols_map(symbols_map_path: str) -> Optional[SymbolsMap]:
  symbols_map = SymbolsMap.load(symbols_map_path) if symbols_map_path else None
  return symbols_map


def save_infer_symbols(prep_dir: str, symbols: List[str]):
  path = os.path.join(prep_dir, INFER_MAP_SYMB_FN)
  save_symbols(path, symbols)


def save_weights_symbols(prep_dir: str, weights_prep_name: str, symbols: List[str]):
  path = os.path.join(prep_dir, f"{weights_prep_name}.symbols")
  save_symbols(path, symbols)


def save_symbols(path: str, symbols: List[str]):
  with open(path, 'w', encoding='utf-8') as f:
    f.write('\n'.join([f"\"{x}\"" for x in symbols]))


def create_or_update_weights_map(base_dir: str, prep_name: str, weights_prep_name: str, template_map: Optional[str] = None):
  prep_dir = get_prepared_dir(base_dir, prep_name)
  assert os.path.isdir(prep_dir)
  orig_prep_dir = get_prepared_dir(base_dir, weights_prep_name)
  assert os.path.isdir(orig_prep_dir)

  logger = init_logger()
  add_console_out_to_logger(logger)
  logger.info(f"Creating/updating weights map for {weights_prep_name}...")

  if template_map is not None:
    _template_map = SymbolsMap.load(template_map)
  else:
    _template_map = None

  if weights_map_exists(prep_dir, weights_prep_name):
    existing_map = load_weights_map(prep_dir, weights_prep_name)
  else:
    existing_map = None

  weights_map, symbols = create_or_update_map(
    orig=load_prep_symbol_converter(orig_prep_dir).get_all_symbols(),
    dest=load_prep_symbol_converter(prep_dir).get_all_symbols(),
    existing_map=existing_map,
    template_map=_template_map,
    logger=logger
  )

  save_weights_map(prep_dir, weights_prep_name, weights_map)
  save_weights_symbols(prep_dir, weights_prep_name, symbols)


def create_or_update_inference_map(base_dir: str, prep_name: str, template_map: Optional[str] = None):
  logger = init_logger()
  add_console_out_to_logger(logger)
  logger.info("Creating/updating inference map...")
  prep_dir = get_prepared_dir(base_dir, prep_name)
  assert os.path.isdir(prep_dir)

  all_symbols = get_all_symbols(prep_dir)

  if template_map is not None:
    _template_map = SymbolsMap.load(template_map)
  else:
    _template_map = None

  if infer_map_exists(prep_dir):
    existing_map = load_infer_map(prep_dir)
  else:
    existing_map = None

  infer_map, symbols = create_or_update_map(
    orig=load_prep_symbol_converter(prep_dir).get_all_symbols(),
    dest=all_symbols,
    existing_map=existing_map,
    template_map=_template_map,
    logger=logger
  )

  save_infer_map(prep_dir, infer_map)
  save_infer_symbols(prep_dir, symbols)


if __name__ == "__main__":
  mode = 2
  if mode == 1:
    create_or_update_weights_map(
      base_dir="/datasets/models/taco2pt_v5",
      prep_name="thchs",
      weights_prep_name="thchs"
    )
  elif mode == 2:
    create_or_update_inference_map(
      base_dir="/datasets/models/taco2pt_v5",
      prep_name="ljs_ipa",
      template_map="maps/weights/thchs_ipa_ljs_ipa.json"
    )
