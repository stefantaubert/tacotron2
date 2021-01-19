import os
from typing import Optional

from src.app.pre.io import (get_text_dir, load_text_symbol_converter,
                            save_text_symbol_converter)
from src.app.pre.mapping import (create_or_update_inference_map,
                                 get_infer_map_path, infer_map_exists)
from src.app.pre.prepare import (get_prepared_dir, load_prep_accents_ids,
                                 load_prep_symbol_converter)
from src.app.utils import prepare_logger
from src.core.common.utils import read_text
from src.core.pre.text.pre_inference import (AccentedSymbol,
                                             AccentedSymbolList,
                                             InferSentenceList, Sentence,
                                             SentenceList)
from src.core.pre.text.pre_inference import add_text as infer_add
from src.core.pre.text.pre_inference import (sents_accent_apply,
                                             sents_accent_template,
                                             sents_convert_to_ipa, sents_map,
                                             sents_normalize, set_accent)
from text_utils import EngToIpaMode, Language, SymbolsMap

_text_csv = "text.csv"
_accents_csv = "accents.csv"


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


def add_text(base_dir: str, prep_name: str, text_name: str, filepath: Optional[str], lang: Language, text: Optional[str]):
  logger = prepare_logger()
  prep_dir = get_prepared_dir(base_dir, prep_name, create=False)
  if not os.path.isdir(prep_dir):
    logger.error("Please prepare data first.")
  else:
    logger.info("Adding text...")
    text_input = ""
    if filepath is None:
      assert text is not None
      text_input = text
    else:
      text_input = read_text(filepath)
    symbol_ids, data = infer_add(
      text=text_input,
      lang=lang,
      logger=logger,
    )
    print("\n" + data.get_formatted(
      symbol_id_dict=symbol_ids,
      accent_id_dict=load_prep_accents_ids(prep_dir)
    ))
    text_dir = get_text_dir(prep_dir, text_name, create=True)
    _save_text_csv(text_dir, data)
    save_text_symbol_converter(text_dir, symbol_ids)
    _accent_template(base_dir, prep_name, text_name)
    _check_for_unknown_symbols(base_dir, prep_name, text_name)


def normalize_text(base_dir: str, prep_name: str, text_name: str):
  logger = prepare_logger()
  prep_dir = get_prepared_dir(base_dir, prep_name, create=False)
  text_dir = get_text_dir(prep_dir, text_name, create=False)
  if not os.path.isdir(text_dir):
    logger.error("Please add text first.")
  else:
    logger.info("Normalizing text...")
    symbol_ids, updated_sentences = sents_normalize(
      sentences=load_text_csv(text_dir),
      text_symbols=load_text_symbol_converter(text_dir),
      logger=logger,
    )
    print("\n" + updated_sentences.get_formatted(
      symbol_id_dict=symbol_ids,
      accent_id_dict=load_prep_accents_ids(prep_dir)
    ))
    _save_text_csv(text_dir, updated_sentences)
    save_text_symbol_converter(text_dir, symbol_ids)
    _accent_template(base_dir, prep_name, text_name)
    _check_for_unknown_symbols(base_dir, prep_name, text_name)


def ipa_convert_text(base_dir: str, prep_name: str, text_name: str, ignore_tones: bool = False, ignore_arcs: bool = True, mode: Optional[EngToIpaMode] = None):
  logger = prepare_logger()
  prep_dir = get_prepared_dir(base_dir, prep_name, create=False)
  text_dir = get_text_dir(prep_dir, text_name, create=False)
  if not os.path.isdir(text_dir):
    logger.error("Please add text first.")
  else:
    logger.info("Converting text to IPA...")
    symbol_ids, updated_sentences = sents_convert_to_ipa(
      sentences=load_text_csv(text_dir),
      text_symbols=load_text_symbol_converter(text_dir),
      ignore_tones=ignore_tones,
      ignore_arcs=ignore_arcs,
      mode=mode,
      logger=logger,
    )
    print("\n" + updated_sentences.get_formatted(
      symbol_id_dict=symbol_ids,
      accent_id_dict=load_prep_accents_ids(prep_dir)
    ))
    _save_text_csv(text_dir, updated_sentences)
    save_text_symbol_converter(text_dir, symbol_ids)
    _accent_template(base_dir, prep_name, text_name)
    _check_for_unknown_symbols(base_dir, prep_name, text_name)


def accent_set(base_dir: str, prep_name: str, text_name: str, accent: str):
  logger = prepare_logger()
  prep_dir = get_prepared_dir(base_dir, prep_name, create=False)
  text_dir = get_text_dir(prep_dir, text_name, create=False)
  if not os.path.isdir(text_dir):
    logger.error("Please add text first.")
  else:
    logger.info(f"Applying accent {accent}...")
    updated_sentences = set_accent(
      sentences=load_text_csv(text_dir),
      accent_ids=load_prep_accents_ids(prep_dir),
      accent=accent
    )
    print("\n" + updated_sentences.get_formatted(
      symbol_id_dict=load_text_symbol_converter(text_dir),
      accent_id_dict=load_prep_accents_ids(prep_dir)
    ))
    _save_text_csv(text_dir, updated_sentences)
    _accent_template(base_dir, prep_name, text_name)
    _check_for_unknown_symbols(base_dir, prep_name, text_name)


def accent_apply(base_dir: str, prep_name: str, text_name: str):
  logger = prepare_logger()
  prep_dir = get_prepared_dir(base_dir, prep_name, create=False)
  text_dir = get_text_dir(prep_dir, text_name, create=False)
  if not os.path.isdir(text_dir):
    logger.error("Please add text first.")
  else:
    logger.info("Applying accents...")
    updated_sentences = sents_accent_apply(
      sentences=load_text_csv(text_dir),
      accented_symbols=_load_accents_csv(text_dir),
      accent_ids=load_prep_accents_ids(prep_dir),
    )
    print("\n" + updated_sentences.get_formatted(
      symbol_id_dict=load_text_symbol_converter(text_dir),
      accent_id_dict=load_prep_accents_ids(prep_dir)
    ))
    _save_text_csv(text_dir, updated_sentences)
    _check_for_unknown_symbols(base_dir, prep_name, text_name)


def map_text(base_dir: str, prep_name: str, text_name: str, symbols_map_path: str, ignore_arcs: bool = True):
  logger = prepare_logger()
  prep_dir = get_prepared_dir(base_dir, prep_name, create=False)
  text_dir = get_text_dir(prep_dir, text_name, create=False)
  if not os.path.isdir(text_dir):
    logger.error("Please add text first.")
  else:
    symbol_ids, updated_sentences = sents_map(
      sentences=load_text_csv(text_dir),
      text_symbols=load_text_symbol_converter(text_dir),
      symbols_map=SymbolsMap.load(symbols_map_path),
      ignore_arcs=ignore_arcs,
      logger=logger,
    )

    print("\n" + updated_sentences.get_formatted(
      symbol_id_dict=symbol_ids,
      accent_id_dict=load_prep_accents_ids(prep_dir)
    ))
    _save_text_csv(text_dir, updated_sentences)
    save_text_symbol_converter(text_dir, symbol_ids)
    _accent_template(base_dir, prep_name, text_name)
    _check_for_unknown_symbols(base_dir, prep_name, text_name)


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
    accented_symbol_list = sents_accent_template(
      sentences=load_text_csv(text_dir),
      text_symbols=load_text_symbol_converter(text_dir),
      accent_ids=load_prep_accents_ids(prep_dir),
    )
    _save_accents_csv(text_dir, accented_symbol_list)


def get_infer_sentences(base_dir: str, prep_name: str, text_name: str) -> InferSentenceList:
  prep_dir = get_prepared_dir(base_dir, prep_name, create=False)
  text_dir = get_text_dir(prep_dir, text_name, create=False)
  if not os.path.isdir(text_dir):
    print(f"The text '{text_name}' doesn't exist.")
    assert False
  result = InferSentenceList.from_sentences(
    sentences=load_text_csv(text_dir),
    accents=load_prep_accents_ids(prep_dir),
    symbols=load_text_symbol_converter(text_dir)
  )

  return result


def _check_for_unknown_symbols(base_dir: str, prep_name: str, text_name: str):
  infer_sents = get_infer_sentences(
    base_dir, prep_name, text_name)

  prep_dir = get_prepared_dir(base_dir, prep_name, create=False)
  logger = prepare_logger()
  unknown_symbols_exist = infer_sents.replace_unknown_symbols(
    model_symbols=load_prep_symbol_converter(prep_dir),
    logger=logger
  )

  if unknown_symbols_exist:
    logger.info(
      "Some symbols are not in the prepared dataset symbolset. You need to create an inference map and then apply it to the symbols.")
  else:
    logger.info("All symbols are in the prepared dataset symbolset. You can now synthesize this text.")
