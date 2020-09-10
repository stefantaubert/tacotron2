import os
from typing import Optional

from src.app.pre.io import (get_text_dir, load_text_symbol_converter,
                            save_text_symbol_converter)
from src.app.pre.mapping import (get_infer_map_path, infer_map_exists,
                                 load_infer_map)
from src.app.pre.prepare import (get_prepared_dir, load_filelist_accents_ids,
                                 load_filelist_symbol_converter)
from src.core.common.language import Language
from src.core.common.symbols_map import SymbolsMap
from src.core.common.utils import read_text
from src.core.pre.text.pre_inference import (AccentedSymbol,
                                             AccentedSymbolList,
                                             InferSentenceList, Sentence,
                                             SentenceList)
from src.core.pre.text.pre_inference import add_text as infer_add
from src.core.pre.text.pre_inference import \
    prepare_for_inference as infer_prepare
from src.core.pre.text.pre_inference import \
    sents_accent_apply as infer_accents_apply
from src.core.pre.text.pre_inference import \
    sents_accent_template as infer_accents_template
from src.core.pre.text.pre_inference import \
    sents_convert_to_ipa as infer_convert_ipa
from src.core.pre.text.pre_inference import sents_map
from src.core.pre.text.pre_inference import sents_normalize as infer_norm
from src.core.pre.text.pre_inference import set_accent

_text_csv = "text.csv"
_accents_csv = "accents.csv"
_inference_csv = "inference.csv"


def load_text_csv(text_dir: str) -> SentenceList:
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


def add_text(base_dir: str, prep_name: str, text_name: str, filepath: str, lang: Language):
  prep_dir = get_prepared_dir(base_dir, prep_name, create=False)
  if not os.path.isdir(prep_dir):
    print("Please prepare data first.")
  else:
    print("Adding text...")
    symbol_ids, data = infer_add(
      text=read_text(filepath),
      lang=lang,
    )
    print("\n" + data.get_formatted(
      symbol_id_dict=symbol_ids,
      accent_id_dict=load_filelist_accents_ids(prep_dir)
    ))
    text_dir = get_text_dir(prep_dir, text_name, create=True)
    _save_text_csv(text_dir, data)
    save_text_symbol_converter(text_dir, symbol_ids)
    _accent_template(base_dir, prep_name, text_name)
    _prepare_inference(base_dir, prep_name, text_name)


def normalize_text(base_dir: str, prep_name: str, text_name: str):
  prep_dir = get_prepared_dir(base_dir, prep_name, create=False)
  text_dir = get_text_dir(prep_dir, text_name, create=False)
  if not os.path.isdir(text_dir):
    print("Please add text first.")
  else:
    print("Normalizing text...")
    symbol_ids, updated_sentences = infer_norm(
      sentences=load_text_csv(text_dir),
      text_symbols=load_text_symbol_converter(text_dir)
    )
    print("\n" + updated_sentences.get_formatted(
      symbol_id_dict=symbol_ids,
      accent_id_dict=load_filelist_accents_ids(prep_dir)
    ))
    _save_text_csv(text_dir, updated_sentences)
    save_text_symbol_converter(text_dir, symbol_ids)
    _accent_template(base_dir, prep_name, text_name)
    _prepare_inference(base_dir, prep_name, text_name)


def ipa_convert_text(base_dir: str, prep_name: str, text_name: str, ignore_tones: bool = False, ignore_arcs: bool = True):
  prep_dir = get_prepared_dir(base_dir, prep_name, create=False)
  text_dir = get_text_dir(prep_dir, text_name, create=False)
  if not os.path.isdir(text_dir):
    print("Please add text first.")
  else:
    print("Converting text to IPA...")
    symbol_ids, updated_sentences = infer_convert_ipa(
      sentences=load_text_csv(text_dir),
      text_symbols=load_text_symbol_converter(text_dir),
      ignore_tones=ignore_tones,
      ignore_arcs=ignore_arcs
    )
    print("\n" + updated_sentences.get_formatted(
      symbol_id_dict=symbol_ids,
      accent_id_dict=load_filelist_accents_ids(prep_dir)
    ))
    _save_text_csv(text_dir, updated_sentences)
    save_text_symbol_converter(text_dir, symbol_ids)
    _accent_template(base_dir, prep_name, text_name)
    _prepare_inference(base_dir, prep_name, text_name)


def accent_set(base_dir: str, prep_name: str, text_name: str, accent: str):
  prep_dir = get_prepared_dir(base_dir, prep_name, create=False)
  text_dir = get_text_dir(prep_dir, text_name, create=False)
  if not os.path.isdir(text_dir):
    print("Please add text first.")
  else:
    print(f"Applying accent {accent}...")
    updated_sentences = set_accent(
      sentences=load_text_csv(text_dir),
      accent_ids=load_filelist_accents_ids(prep_dir),
      accent=accent
    )
    print("\n" + updated_sentences.get_formatted(
      symbol_id_dict=load_text_symbol_converter(text_dir),
      accent_id_dict=load_filelist_accents_ids(prep_dir)
    ))
    _save_text_csv(text_dir, updated_sentences)
    _accent_template(base_dir, prep_name, text_name)
    _prepare_inference(base_dir, prep_name, text_name)


def accent_apply(base_dir: str, prep_name: str, text_name: str):
  prep_dir = get_prepared_dir(base_dir, prep_name, create=False)
  text_dir = get_text_dir(prep_dir, text_name, create=False)
  if not os.path.isdir(text_dir):
    print("Please add text first.")
  else:
    print("Applying accents...")
    updated_sentences = infer_accents_apply(
      sentences=load_text_csv(text_dir),
      accented_symbols=_load_accents_csv(text_dir),
      accent_ids=load_filelist_accents_ids(prep_dir),
    )
    print("\n" + updated_sentences.get_formatted(
      symbol_id_dict=load_text_symbol_converter(text_dir),
      accent_id_dict=load_filelist_accents_ids(prep_dir)
    ))
    _save_text_csv(text_dir, updated_sentences)
    _prepare_inference(base_dir, prep_name, text_name)


def map_text(base_dir: str, prep_name: str, text_name: str, symbols_map_path: str, ignore_arcs: bool = True):
  prep_dir = get_prepared_dir(base_dir, prep_name, create=False)
  text_dir = get_text_dir(prep_dir, text_name, create=False)
  if not os.path.isdir(text_dir):
    print("Please add text first.")
  else:
    symbol_ids, updated_sentences = sents_map(
      sentences=load_text_csv(text_dir),
      text_symbols=load_text_symbol_converter(text_dir),
      symbols_map=SymbolsMap.load(symbols_map_path),
      ignore_arcs=ignore_arcs
    )

    print("\n" + updated_sentences.get_formatted(
      symbol_id_dict=symbol_ids,
      accent_id_dict=load_filelist_accents_ids(prep_dir)
    ))
    _save_text_csv(text_dir, updated_sentences)
    save_text_symbol_converter(text_dir, symbol_ids)
    _accent_template(base_dir, prep_name, text_name)
    _prepare_inference(base_dir, prep_name, text_name)


def map_to_prep_symbols(base_dir: str, prep_name: str, text_name: str, ignore_arcs: bool = True):
  prep_dir = get_prepared_dir(base_dir, prep_name, create=False)
  assert os.path.isdir(prep_dir)
  assert infer_map_exists(prep_dir)

  symb_map_path = get_infer_map_path(prep_dir)
  map_text(base_dir, prep_name, text_name, symb_map_path, ignore_arcs)


def _accent_template(base_dir: str, prep_name: str, text_name: str):
  prep_dir = get_prepared_dir(base_dir, prep_name, create=False)
  text_dir = get_text_dir(prep_dir, text_name, create=False)
  if not os.path.isdir(text_dir):
    print("Please add text first.")
  else:
    print("Updating accent template...")
    accented_symbol_list = infer_accents_template(
      sentences=load_text_csv(text_dir),
      text_symbols=load_text_symbol_converter(text_dir),
      accent_ids=load_filelist_accents_ids(prep_dir),
    )
    _save_accents_csv(text_dir, accented_symbol_list)


def _prepare_inference(base_dir: str, prep_name: str, text_name: str):
  prep_dir = get_prepared_dir(base_dir, prep_name, create=False)
  text_dir = get_text_dir(prep_dir, text_name, create=False)
  if not os.path.isdir(text_dir):
    print("Please add text first.")
  else:
    print("Updating text for inference...")
    symbs = load_filelist_symbol_converter(prep_dir)
    infer_sents, unknown_symbols_exist = infer_prepare(
      sentences=load_text_csv(text_dir),
      text_symbols=load_text_symbol_converter(text_dir),
      known_symbols=symbs
    )
    print("\n" + infer_sents.get_formatted(
      symbol_id_dict=symbs,
      accent_id_dict=load_filelist_accents_ids(prep_dir)
    ))
    if unknown_symbols_exist:
      print("Some symbols are not in the prepared dataset symbolset. You need to create an inference map and then apply it to the symbols.")
    _save_inference_csv(text_dir, infer_sents)


if __name__ == "__main__":
  mode = 4
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

    map_to_prep_symbols(
      base_dir="/datasets/models/taco2pt_v5",
      prep_name="thchs_ljs",
      text_name="north"
    )

  elif mode == 2:
    accent_apply(
      base_dir="/datasets/models/taco2pt_v5",
      prep_name="thchs_ljs",
      text_name="north",
    )
  elif mode == 3:
    add_text(
      base_dir="/datasets/models/taco2pt_v5",
      prep_name="ljs_ipa",
      text_name="ipa-north_sven_orig",
      filepath="examples/ipa/north_sven_orig.txt",
      lang=Language.IPA,
    )

    normalize_text(
      base_dir="/datasets/models/taco2pt_v5",
      prep_name="ljs_ipa",
      text_name="ipa-north_sven_orig",
    )

  elif mode == 4:
    add_text(
      base_dir="/datasets/models/taco2pt_v5",
      prep_name="ljs_ipa",
      text_name="en-coma",
      filepath="examples/en/coma.txt",
      lang=Language.ENG,
    )

    normalize_text(
      base_dir="/datasets/models/taco2pt_v5",
      prep_name="ljs_ipa",
      text_name="en-coma",
    )

    ipa_convert_text(
      base_dir="/datasets/models/taco2pt_v5",
      prep_name="ljs_ipa",
      text_name="en-coma",
    )

    map_to_prep_symbols(
      base_dir="/datasets/models/taco2pt_v5",
      prep_name="ljs_ipa",
      text_name="en-coma",
    )
