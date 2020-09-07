import os
from typing import Optional

from src.core.common import get_subdir, read_text, Language, SymbolIdDict, SymbolsMap
from src.core.pre import Sentence, SentenceList, infer_add, infer_convert_ipa, infer_map, infer_norm, infer_accents_template, infer_accents_apply, AccentedSymbolList, AccentedSymbol, InferSentenceList, infer_prepare
from src.app.pre.prepare import get_prepared_dir, load_filelist_accents_ids, load_filelist_symbol_converter

_text_csv = "text.csv"
_accents_csv = "accents.csv"
_inference_csv = "inference.csv"
_symbols_json = "symbols.json"


def get_text_dir(prep_dir: str, text_name: str, create: bool):
  return get_subdir(prep_dir, text_name, create)


def _load_text_csv(text_dir: str) -> SentenceList:
  path = os.path.join(text_dir, _text_csv)
  return SentenceList.load(Sentence, path)


def _save_text_csv(text_dir: str, data: SentenceList):
  path = os.path.join(text_dir, _text_csv)
  data.save(path)


def _load_accents_csv(text_dir: str) -> AccentedSymbolList:
  path = os.path.join(text_dir, _accents_csv)
  return AccentedSymbolList.load(AccentedSymbol, path)


def _save_accents_csv(text_dir: str, data: AccentedSymbolList):
  path = os.path.join(text_dir, _accents_csv)
  data.save(path)


def load_inference_csv(text_dir: str) -> InferSentenceList:
  path = os.path.join(text_dir, _inference_csv)
  return InferSentenceList.load(Sentence, path)


def _save_inference_csv(text_dir: str, data: InferSentenceList):
  path = os.path.join(text_dir, _inference_csv)
  data.save(path)


def _load_text_symbol_converter(text_dir: str) -> SymbolIdDict:
  path = os.path.join(text_dir, _symbols_json)
  return SymbolIdDict.load_from_file(path)


def _save_text_symbol_converter(text_dir: str, data: SymbolIdDict):
  path = os.path.join(text_dir, _symbols_json)
  data.save(path)


def add_text(base_dir: str, prep_name: str, text_name: str, filepath: str, lang: Language, accent: Optional[str] = None, replace_unknown_ipa_by: Optional[str] = "_"):
  prep_dir = get_prepared_dir(base_dir, prep_name, create=False)
  if not os.path.isdir(prep_dir):
    print("Please prepare data first.")
  else:
    print("Adding text...")
    symbol_ids, data = infer_add(
      text=read_text(filepath),
      accent_ids=load_filelist_accents_ids(prep_dir),
      lang=lang,
      accent=accent,
      replace_unknown_ipa_by=replace_unknown_ipa_by
    )
    text_dir = get_text_dir(prep_dir, text_name, create=True)
    _save_text_csv(text_dir, data)
    _save_text_symbol_converter(text_dir, symbol_ids)
    _accent_template(base_dir, prep_name, text_name)
    _prepare_inference(base_dir, prep_name, text_name, replace_unknown_ipa_by)


def normalize_text(base_dir: str, prep_name: str, text_name: str, replace_unknown_ipa_by: str = "_"):
  prep_dir = get_prepared_dir(base_dir, prep_name, create=False)
  text_dir = get_text_dir(prep_dir, text_name, create=False)
  if not os.path.isdir(text_dir):
    print("Please add text first.")
  else:
    print("Normalizing text...")
    symbol_ids, updated_sentences = infer_norm(
      sentences=_load_text_csv(text_dir),
      text_symbols=_load_text_symbol_converter(text_dir)
    )
    _save_text_csv(text_dir, updated_sentences)
    _save_text_symbol_converter(text_dir, symbol_ids)
    _accent_template(base_dir, prep_name, text_name)
    _prepare_inference(base_dir, prep_name, text_name, replace_unknown_ipa_by)


def ipa_convert_text(base_dir: str, prep_name: str, text_name: str, ignore_tones: bool = False, ignore_arcs: bool = True, replace_unknown_ipa_by: str = "_"):
  prep_dir = get_prepared_dir(base_dir, prep_name, create=False)
  text_dir = get_text_dir(prep_dir, text_name, create=False)
  if not os.path.isdir(text_dir):
    print("Please add text first.")
  else:
    print("Converting text to IPA...")
    symbol_ids, updated_sentences = infer_convert_ipa(
      sentences=_load_text_csv(text_dir),
      text_symbols=_load_text_symbol_converter(text_dir),
      ignore_tones=ignore_tones,
      ignore_arcs=ignore_arcs,
      replace_unknown_ipa_by=replace_unknown_ipa_by
    )
    _save_text_csv(text_dir, updated_sentences)
    _save_text_symbol_converter(text_dir, symbol_ids)
    _accent_template(base_dir, prep_name, text_name)
    _prepare_inference(base_dir, prep_name, text_name, replace_unknown_ipa_by)


def accent_apply(base_dir: str, prep_name: str, text_name: str, replace_unknown_ipa_by: str = "_"):
  prep_dir = get_prepared_dir(base_dir, prep_name, create=False)
  text_dir = get_text_dir(prep_dir, text_name, create=False)
  if not os.path.isdir(text_dir):
    print("Please add text first.")
  else:
    print("Applying accents...")
    updated_sentences = infer_accents_apply(
      sentences=_load_text_csv(text_dir),
      accented_symbols=_load_accents_csv(text_dir),
      accent_ids=load_filelist_accents_ids(prep_dir),
    )
    _save_text_csv(text_dir, updated_sentences)
    _prepare_inference(base_dir, prep_name, text_name, replace_unknown_ipa_by)


def map_text(base_dir: str, prep_name: str, text_name: str, symbols_map: str, replace_unknown_ipa_by: str = "_"):
  prep_dir = get_prepared_dir(base_dir, prep_name, create=False)
  text_dir = get_text_dir(prep_dir, text_name, create=False)
  if not os.path.isdir(text_dir):
    print("Please add text first.")
  else:
    symbol_ids, updated_sentences = infer_map(
      sentences=_load_text_csv(text_dir),
      symbols_map=SymbolsMap.load(symbols_map)
    )
    _save_text_csv(text_dir, updated_sentences)
    _save_text_symbol_converter(text_dir, symbol_ids)
    _accent_template(base_dir, prep_name, text_name)
    _prepare_inference(base_dir, prep_name, text_name, replace_unknown_ipa_by)


def map_to_prep_symbols(base_dir: str, prep_name: str, text_name: str, replace_unknown_ipa_by: str = "_"):
  #prep_dir = get_prepared_dir(base_dir, prep_name, create=False)
  # TODO load map
  symbols_map = ""
  map_text(base_dir, prep_name, text_name, symbols_map, replace_unknown_ipa_by)


def _accent_template(base_dir: str, prep_name: str, text_name: str):
  prep_dir = get_prepared_dir(base_dir, prep_name, create=False)
  text_dir = get_text_dir(prep_dir, text_name, create=False)
  if not os.path.isdir(text_dir):
    print("Please add text first.")
  else:
    print("Updating accent template...")
    accented_symbol_list = infer_accents_template(
      sentences=_load_text_csv(text_dir),
      text_symbols=_load_text_symbol_converter(text_dir),
      accent_ids=load_filelist_accents_ids(prep_dir),
    )
    _save_accents_csv(text_dir, accented_symbol_list)


def _prepare_inference(base_dir: str, prep_name: str, text_name: str, replace_unknown_ipa_by: str = "_"):
  prep_dir = get_prepared_dir(base_dir, prep_name, create=False)
  text_dir = get_text_dir(prep_dir, text_name, create=False)
  if not os.path.isdir(text_dir):
    print("Please add text first.")
  else:
    print("Updating text for inference...")
    infer_sents = infer_prepare(
      sentences=_load_text_csv(text_dir),
      text_symbols=_load_text_symbol_converter(text_dir),
      known_symbols=load_filelist_symbol_converter(prep_dir),
      replace_unknown_ipa_by=replace_unknown_ipa_by
    )
    _save_inference_csv(text_dir, infer_sents)


if __name__ == "__main__":
  mode = 1
  if mode == 1:
    add_text(
      base_dir="/datasets/models/taco2pt_v5",
      prep_name="thchs_ljs",
      text_name="north",
      filepath="examples/en/north.txt",
      lang=Language.ENG,
    )

    normalize_text(
      base_dir="/datasets/models/taco2pt_v5",
      prep_name="thchs_ljs",
      text_name="north",
    )

    ipa_convert_text(
      base_dir="/datasets/models/taco2pt_v5",
      prep_name="thchs_ljs",
      text_name="north",
    )

    # map_text(
    #   base_dir="/datasets/models/taco2pt_v5",
    #   prep_name="thchs_ljs",
    #   text_name="north",
    #   symbols_map="",
    # )

  elif mode == 2:
    accent_apply(
      base_dir="/datasets/models/taco2pt_v5",
      prep_name="thchs_ljs",
      text_name="north",
    )
