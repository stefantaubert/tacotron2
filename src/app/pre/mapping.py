import os
from src.app.pre.prepare import get_prepared_dir, load_filelist_symbol_converter
from src.core.pre import create_weights_map as create_weights_map_core, SymbolsMap, create_inference_map as create_inference_map_core
from typing import List, Optional

def load_weights_map(symbols_map_path: str) -> Optional[SymbolsMap]:
  symbols_map = SymbolsMap.load(symbols_map_path) if symbols_map_path else None
  return symbols_map

def _save_weights_map(dest_dir: str, dest_prep_name: str, orig_prep_name: str, weights_map: SymbolsMap):
  path = os.path.join(dest_dir, f"{dest_prep_name}_{orig_prep_name}.json")
  weights_map.save(path)

def _save_infer_map(dest_dir: str, prep_name: str, infer_map: SymbolsMap):
  path = os.path.join(dest_dir, f"{prep_name}.json")
  infer_map.save(path)

def _save_infer_symbols(dest_dir: str, prep_name: str, symbols: List[str]):
  path = os.path.join(dest_dir, f"{prep_name}.symbols")
  with open(path, 'w', encoding='utf-8') as f:
    f.write('\n'.join([f"\"{x}\"" for x in symbols]))

def _save_weights_symbols(dest_dir: str, dest_prep_name: str, orig_prep_name: str, symbols: List[str]):
  path = os.path.join(dest_dir, f"{dest_prep_name}_{orig_prep_name}.symbols")
  with open(path, 'w', encoding='utf-8') as f:
    f.write('\n'.join([f"\"{x}\"" for x in symbols]))

def _read_corpora(path: str) -> str:
  with open(path, 'r', encoding='utf-8') as f:
    content = ''.join(f.readlines())
  return content

def create_weights_map(base_dir: str, dest_prep_name: str, orig_prep_name: str, dest_dir: str = "maps/weights"):
  dest_prep_dir = get_prepared_dir(base_dir, dest_prep_name)
  orig_prep_dir = get_prepared_dir(base_dir, orig_prep_name)
  assert os.path.isdir(dest_prep_dir)
  assert os.path.isdir(orig_prep_dir)
  dest_conv = load_filelist_symbol_converter(dest_prep_dir)
  orig_conv = load_filelist_symbol_converter(orig_prep_dir)
  weights_map, symbols = create_weights_map_core(orig_conv, dest_conv)
  _save_weights_map(dest_dir, dest_prep_name, orig_prep_name, weights_map)
  _save_weights_symbols(dest_dir, dest_prep_name, orig_prep_name, symbols)

def create_inference_map(base_dir: str, prep_name: str, corpora: str, is_ipa: bool = False, ignore_tones: bool = False, ignore_arcs: bool = True, existing_map: Optional[str] = None, dest_dir: str = "maps/inference"):
  assert os.path.isfile(corpora)
  prep_dir = get_prepared_dir(base_dir, prep_name)
  assert os.path.isdir(prep_dir)
  model_conv = load_filelist_symbol_converter(prep_dir)
  corpora_content = _read_corpora(corpora)
  existing_map = load_weights_map(existing_map)
  infer_map, symbols = create_inference_map_core(model_conv, corpora_content, is_ipa, ignore_tones, ignore_arcs, existing_map=existing_map)
  _save_infer_map(dest_dir, prep_name, infer_map)
  _save_infer_symbols(dest_dir, prep_name, symbols)

if __name__ == "__main__":
  mode = 2
  if mode == 1:
    create_weights_map(
      base_dir="/datasets/models/taco2pt_v3",
      dest_prep_name="thchs",
      orig_prep_name="thchs",
    )
  elif mode == 2:
    create_inference_map(
      base_dir="/datasets/models/taco2pt_v3",
      prep_name="thchs",
      corpora="examples/ipa/corpora.txt",
      existing_map="maps/inference/chn_v1.json",
      is_ipa=True,
    )