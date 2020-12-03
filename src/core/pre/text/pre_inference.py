from dataclasses import dataclass
from logging import Logger
from math import ceil
from typing import List, Optional, Set, Tuple

from src.core.common.accents_dict import AccentsDict
from src.core.common.globals import PADDING_SYMBOL
from src.core.common.language import Language
from src.core.common.symbol_id_dict import SymbolIdDict
from src.core.common.symbols_map import SymbolsMap
from src.core.common.text import deserialize_list, serialize_list
from src.core.common.utils import (GenericList, console_out_len,
                                   get_unique_items)
from src.core.pre.text.utils import symbols_convert_to_ipa, symbols_normalize
from text_utils.ipa2symb import IPAExtractionSettings
from text_utils.text import EngToIpaMode, text_to_sentences, text_to_symbols


def get_formatted_core(sent_id: int, symbols: List[str], accent_ids: List[int], max_pairs_per_line: int, space_length: int, accent_id_dict: AccentsDict) -> str:
  assert len(symbols) == len(accent_ids)
  final_symbols: List[str] = []
  final_accent_ids: List[str] = []
  for symbol, accent_id in zip(symbols, accent_ids):
    max_width = max(console_out_len(symbol), len(str(accent_id)))
    symbol_str = symbol + " " * (max_width - console_out_len(symbol))
    accent_id_str = str(accent_id) + " " * (max_width - len(str(accent_id)))
    final_symbols.append(symbol_str)
    final_accent_ids.append(accent_id_str)
  count_lines = ceil(len(final_symbols) / max_pairs_per_line)
  batches = []
  for i in range(count_lines):
    start_idx = i * max_pairs_per_line
    end = min((i + 1) * max_pairs_per_line, len(final_symbols))
    batches.append((final_symbols[start_idx:end], final_accent_ids[start_idx:end]))
  final_pair_lines = []
  for batch_symbs, batch_acc_ids in batches:
    symbols_line = (" " * space_length).join(batch_symbs)
    accents_line = (" " * space_length).join(batch_acc_ids)
    final_pair_lines.append(symbols_line)
    final_pair_lines.append(accents_line)
    accent_ids_list = []
    for occuring_accent_id in sorted({int(x) for x in batch_acc_ids}):
      accent_ids_list.append(
        f"{occuring_accent_id}={accent_id_dict.get_accent(occuring_accent_id)}")
    accends_names_lines = ', '.join(accent_ids_list)
    final_pair_lines.append(accends_names_lines)
    # final_pair_lines.append("")
  # add symbols count
  final_pair_lines[-3] += f" ({len(final_symbols)})"
  # final_pair_lines[-4] += f" ({len(final_symbols)})"
  sent_mark = f"{sent_id}: "
  final_pair_lines[0] = f"{sent_mark}{final_pair_lines[0]}"
  for i in range(len(final_pair_lines) - 1):
    final_pair_lines[i + 1] = " " * len(sent_mark) + final_pair_lines[i + 1]

  result = "\n".join(final_pair_lines)
  return result


# def get_right_nearest_index_of_symbol(text: str, position: int, symbol: str) -> int:
#   """returns -1 if symbol not in <= position in text, otherwise the first index"""
#   assert position < len(text)
#   assert len(symbol) == 1
#   while position >= 0:
#     if text[position] != symbol:
#       position -= 1
#   return position


@dataclass()
class Sentence:
  sent_id: int
  text: str
  lang: Language
  serialized_symbols: str
  serialized_accents: str

  def get_symbol_ids(self):
    return deserialize_list(self.serialized_symbols)

  def get_accent_ids(self):
    return deserialize_list(self.serialized_accents)

  def get_formatted(self, symbol_id_dict: SymbolIdDict, accent_id_dict: AccentsDict, pairs_per_line=170, space_length=0):
    return get_formatted_core(
      sent_id=self.sent_id,
      symbols=symbol_id_dict.get_symbols(self.serialized_symbols),
      accent_ids=self.get_accent_ids(),
      accent_id_dict=accent_id_dict,
      space_length=space_length,
      max_pairs_per_line=pairs_per_line
    )


class SentenceList(GenericList[Sentence]):
  # def get_occuring_symbols(self) -> Set[str]:
  #   ipa_settings = IPAExtractionSettings(
  #     ignore_tones=False,
  #     ignore_arcs=False,
  #     replace_unknown_ipa_by=PADDING_SYMBOL,
  #   )

  #   return get_unique_items([text_to_symbols(
  #     text=x.text,
  #     lang=x.lang,
  #     ipa_settings=ipa_settings,
  #     ) for x in self.items()])

  def get_formatted(self, symbol_id_dict: SymbolIdDict, accent_id_dict: AccentsDict):
    result = "\n".join([sentence.get_formatted(symbol_id_dict, accent_id_dict)
                        for sentence in self.items()])
    return result


@dataclass()
class AccentedSymbol:
  position: str
  symbol: str
  accent: str


class AccentedSymbolList(GenericList[AccentedSymbol]):
  pass


@dataclass
class InferSentence:
  sent_id: int
  symbols: List[str]
  accents: List[str]

  def get_formatted(self, accent_id_dict: AccentsDict, pairs_per_line=170, space_length=0):
    return get_formatted_core(
      sent_id=self.sent_id,
      symbols=self.symbols,
      accent_ids=accent_id_dict.get_ids(self.accents),
      accent_id_dict=accent_id_dict,
      space_length=space_length,
      max_pairs_per_line=pairs_per_line
    )


class InferSentenceList(GenericList[InferSentence]):
  @classmethod
  def from_sentences(cls, sentences: SentenceList, accents: AccentsDict, symbols: SymbolIdDict):
    res = cls()
    for sentence in sentences.items():
      infer_sent = InferSentence(
        sent_id=sentence.sent_id,
        symbols=symbols.get_symbols(sentence.serialized_symbols),
        accents=accents.get_accents(sentence.serialized_accents)
      )
      assert len(infer_sent.symbols) == len(infer_sent.accents)
      res.append(infer_sent)
    return res

  def replace_unknown_symbols(self, model_symbols: SymbolIdDict, logger: Logger) -> bool:
    unknown_symbols_exist = False
    for sentence in self.items():
      if model_symbols.has_unknown_symbols(sentence.symbols):
        sentence.symbols = model_symbols.replace_unknown_symbols_with_pad(sentence.symbols)
        text = SymbolIdDict.symbols_to_text(sentence.symbols)
        logger.info(f"Sentence {sentence.sent_id} contains unknown symbols: {text}")
        unknown_symbols_exist = True
        assert len(sentence.symbols) == len(sentence.accents)
    return unknown_symbols_exist

  def to_sentence(self, space_symbol: str, space_accent: str) -> InferSentence:
    res = InferSentence(
      sent_id=1,
      symbols=[],
      accents=[],
    )

    for sent in self.items():
      res.symbols.extend(sent.symbols + [space_symbol])
      res.accents.extend(sent.accents + [space_accent])

    return res


def add_text(text: str, lang: Language, logger: Logger) -> Tuple[SymbolIdDict, SentenceList]:
  res = SentenceList()
  sents = text_to_sentences(
    text=text,
    lang=lang,
    logger=logger,
  )

  default_accent_id = 0
  ipa_settings = IPAExtractionSettings(
    ignore_tones=False,
    ignore_arcs=False,
    replace_unknown_ipa_by=PADDING_SYMBOL,
  )

  sents_symbols: List[List[str]] = [text_to_symbols(
    sent,
    lang=lang,
    ipa_settings=ipa_settings,
    logger=logger,
  ) for sent in sents]
  symbols = SymbolIdDict.init_from_symbols(get_unique_items(sents_symbols))
  for i, sent_symbols in enumerate(sents_symbols):
    sentence = Sentence(
      sent_id=i + 1,
      lang=lang,
      serialized_symbols=symbols.get_serialized_ids(sent_symbols),
      serialized_accents=serialize_list([default_accent_id] * len(sent_symbols)),
      text=SymbolIdDict.symbols_to_text(sent_symbols),
    )
    res.append(sentence)
  return symbols, res


def set_accent(sentences: SentenceList, accent_ids: AccentsDict, accent: str) -> Tuple[SymbolIdDict, SentenceList]:
  accent_id = accent_ids.get_id(accent)
  for sentence in sentences.items():
    new_accent_ids = [accent_id] * len(sentence.get_accent_ids())
    sentence.serialized_accents = serialize_list(new_accent_ids)
    assert len(sentence.get_accent_ids()) == len(sentence.get_symbol_ids())
  return sentences


def sents_normalize(sentences: SentenceList, text_symbols: SymbolIdDict, logger: Logger) -> Tuple[SymbolIdDict, SentenceList]:
  # Maybe add info if something was unknown
  sents_new_symbols = []
  for sentence in sentences.items():
    new_symbols, new_accent_ids = symbols_normalize(
      symbols=text_symbols.get_symbols(sentence.serialized_symbols),
      lang=sentence.lang,
      accent_ids=deserialize_list(sentence.serialized_accents),
      logger=logger,
    )
    # TODO: check if new sentences resulted and then split them.
    sentence.serialized_accents = serialize_list(new_accent_ids)
    sents_new_symbols.append(new_symbols)

  return update_symbols_and_text(sentences, sents_new_symbols)


def update_symbols_and_text(sentences: SentenceList, sents_new_symbols: List[List[str]]):
  symbols = SymbolIdDict.init_from_symbols(get_unique_items([x for x in sents_new_symbols]))
  for sentence, new_symbols in zip(sentences.items(), sents_new_symbols):
    sentence.serialized_symbols = symbols.get_serialized_ids(new_symbols)
    sentence.text = SymbolIdDict.symbols_to_text(new_symbols)
    assert len(sentence.get_symbol_ids()) == len(new_symbols)
    assert len(sentence.get_accent_ids()) == len(new_symbols)
  return symbols, sentences


def sents_convert_to_ipa(sentences: SentenceList, text_symbols: SymbolIdDict, ignore_tones: bool, ignore_arcs: bool, mode: Optional[EngToIpaMode], logger: Logger) -> Tuple[SymbolIdDict, SentenceList]:

  sents_new_symbols = []
  for sentence in sentences.items(True):
    if sentence.lang == Language.ENG and mode is None:
      ex = "Please specify the ipa conversion mode."
      logger.exception(ex)
      raise Exception(ex)
    new_symbols, new_accent_ids = symbols_convert_to_ipa(
      symbols=text_symbols.get_symbols(sentence.serialized_symbols),
      lang=sentence.lang,
      accent_ids=deserialize_list(sentence.serialized_accents),
      ignore_arcs=ignore_arcs,
      ignore_tones=ignore_tones,
      mode=mode,
      replace_unknown_with=PADDING_SYMBOL,
      logger=logger,
    )
    assert len(new_symbols) == len(new_accent_ids)
    sentence.lang = Language.IPA
    sentence.serialized_accents = serialize_list(new_accent_ids)
    sents_new_symbols.append(new_symbols)
    assert len(sentence.get_accent_ids()) == len(new_symbols)

  return update_symbols_and_text(sentences, sents_new_symbols)


def sents_map(sentences: SentenceList, text_symbols: SymbolIdDict, symbols_map: SymbolsMap, ignore_arcs: bool, logger: Logger) -> Tuple[SymbolIdDict, SentenceList]:
  sents_new_symbols = []
  result = SentenceList()
  new_sent_id = 0

  ipa_settings = IPAExtractionSettings(
    ignore_tones=False,
    ignore_arcs=ignore_arcs,
    replace_unknown_ipa_by=PADDING_SYMBOL,
  )

  for sentence in sentences.items():
    symbols = text_symbols.get_symbols(sentence.serialized_symbols)
    accent_ids = deserialize_list(sentence.serialized_accents)

    mapped_symbols = symbols_map.apply_to_symbols(symbols)

    text = SymbolIdDict.symbols_to_text(mapped_symbols)
    # a resulting empty text would make no problems
    sents = text_to_sentences(
      text=text,
      lang=sentence.lang,
      logger=logger,
    )

    for new_sent_text in sents:
      new_symbols = text_to_symbols(
        new_sent_text,
        lang=sentence.lang,
        ipa_settings=ipa_settings,
        logger=logger,
      )

      if len(accent_ids) > 0:
        new_accent_ids = [accent_ids[0]] * len(new_symbols)
      else:
        new_accent_ids = []

      assert len(new_accent_ids) == len(new_symbols)

      new_sent_id += 1
      tmp = Sentence(
        sent_id=new_sent_id,
        text=new_sent_text,
        lang=sentence.lang,
        serialized_accents=serialize_list(new_accent_ids),
        serialized_symbols=""
      )
      sents_new_symbols.append(new_symbols)

      assert len(tmp.get_accent_ids()) == len(new_symbols)
      result.append(tmp)

  return update_symbols_and_text(result, sents_new_symbols)


# def sents_rules(sentences: SentenceList, rules: str) -> SentenceList:
#   pass


def sents_accent_template(sentences: SentenceList, text_symbols: SymbolIdDict, accent_ids: AccentsDict) -> AccentedSymbolList:
  res = AccentedSymbolList()
  for i, sent in enumerate(sentences.items()):
    symbols = text_symbols.get_symbols(sent.serialized_symbols)
    accents = accent_ids.get_accents(sent.serialized_accents)
    for j, symbol_accent in enumerate(zip(symbols, accents)):
      symbol, accent = symbol_accent
      accented_symbol = AccentedSymbol(
        position=f"{i}-{j}",
        symbol=symbol,
        accent=accent
      )
      res.append(accented_symbol)
  return res


def sents_accent_apply(sentences: SentenceList, accented_symbols: AccentedSymbolList, accent_ids: AccentsDict) -> SentenceList:
  current_index = 0
  for sent in sentences.items():
    accent_ids_count = len(deserialize_list(sent.serialized_accents))
    assert len(accented_symbols) >= current_index + accent_ids_count
    accented_symbol_selection: List[AccentedSymbol] = accented_symbols[current_index:current_index + accent_ids_count]
    current_index += accent_ids_count
    new_accent_ids = accent_ids.get_ids([x.accent for x in accented_symbol_selection])
    sent.serialized_accents = serialize_list(new_accent_ids)
    assert len(sent.get_accent_ids()) == len(sent.get_symbol_ids())
  return sentences


def prepare_for_inference(sentences: SentenceList, text_symbols: SymbolIdDict, text_accents: AccentsDict, known_symbols: SymbolIdDict, logger: Logger) -> InferSentenceList:
  result = InferSentenceList.from_sentences(sentences, text_accents, text_symbols)
  unknown_exist = result.replace_unknown_symbols(known_symbols, logger)
  return result, unknown_exist
