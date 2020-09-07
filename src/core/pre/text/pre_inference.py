from src.core.pre.text.utils import symbols_convert_to_ipa, symbols_normalize
from src.core.common.ipa2symb import extract_from_sentence
from src.core.common import serialize_list
from src.core.common import AccentsDict
from src.core.common import GenericList
from typing import List, Optional, Set, Tuple
from src.core.common import Language, convert_to_ipa, text_to_symbols, split_sentences, normalize, get_unique_items, deserialize_list, serialize_list, text_to_symbols, convert_to_ipa as text_convert_to_ipa, normalize as text_normalize, get_unique_items, get_counter
from dataclasses import dataclass
from src.core.common import SymbolIdDict, SymbolsMap


@dataclass()
class Sentence:
  sent_id: int
  text: str
  lang: Language
  serialized_symbols: str
  serialized_accents: str
  # Contains only the symbols which are known

  def get_symbol_ids(self):
    return deserialize_list(self.serialized_symbols)

  def get_accent_ids(self):
    return deserialize_list(self.serialized_accents)

  def get_formatted(self, symbol_id_dict: SymbolIdDict, accent_id_dict: AccentsDict):
    part1 = f"{self.sent_id}: "
    result = f"{part1}{symbol_id_dict.get_text(self.serialized_symbols)} ({len(self.get_symbol_ids())})\n"
    result += f"{' ' * len(part1)}{''.join(map(str, self.get_accent_ids()))}\n"
    accent_ids_list = []
    for occuring_accent_id in sorted(set(self.get_accent_ids())):
      accent_ids_list.append(
        f"{occuring_accent_id} = {accent_id_dict.get_accent(occuring_accent_id)}")
    result += f"{' ' * len(part1)}{', '.join(accent_ids_list)}"
    return result


class SentenceList(GenericList[Sentence]):
  def get_occuring_symbols(self) -> Set[str]:
    return get_unique_items([text_to_symbols(x.text, x.lang) for x in self.items()])

  def get_formatted(self, symbol_id_dict: SymbolIdDict, accent_id_dict: AccentsDict):
    result = ""
    for sentence in self.items():
      result += sentence.get_formatted(symbol_id_dict, accent_id_dict) + "\n"
    return result


class InferSentenceList(SentenceList):
  pass


@dataclass()
class AccentedSymbol:
  position: str
  symbol: str
  accent: str


class AccentedSymbolList(GenericList[AccentedSymbol]):
  pass


def add_text(text: str, lang: Language, accent_ids: AccentsDict, accent: Optional[str]) -> Tuple[SymbolIdDict, SentenceList]:
  res = SentenceList()
  sents = split_sentences(text, lang)
  accent_id = accent_ids.get_id(accent) if accent is not None else 0
  sents_symbols: List[List[str]] = [text_to_symbols(
    sent,
    lang=lang,
    ignore_tones=False,
    ignore_arcs=False
  ) for sent in sents]
  symbols = SymbolIdDict.init_from_symbols(get_unique_items(sents_symbols))
  for i, sent_symbols in enumerate(sents_symbols):
    sentence = Sentence(
      sent_id=i,
      lang=lang,
      serialized_symbols=symbols.get_serialized_ids(sent_symbols),
      serialized_accents=serialize_list([accent_id] * len(sent_symbols)),
      text=SymbolIdDict.symbols_to_str(sent_symbols),
    )
    res.append(sentence)
  return symbols, res


def sents_normalize(sentences: SentenceList, text_symbols: SymbolIdDict) -> Tuple[SymbolIdDict, SentenceList]:
  # Maybe add info if something was unknown
  sents_new_symbols = []
  for sentence in sentences.items():
    new_symbols, new_accent_ids = symbols_normalize(
      symbols=text_symbols.get_symbols(sentence.serialized_symbols),
      lang=sentence.lang,
      accent_ids=deserialize_list(sentence.serialized_accents)
    )
    sentence.serialized_accents = serialize_list(new_accent_ids)
    sents_new_symbols.append(new_symbols)

  return update_symbols_and_text(sentences, sents_new_symbols)


def update_symbols_and_text(sentences: SentenceList, sents_new_symbols: List[List[str]]):
  symbols = SymbolIdDict.init_from_symbols(get_unique_items([x for x in sents_new_symbols]))
  for sentence, new_symbols in zip(sentences.items(), sents_new_symbols):
    sentence.serialized_symbols = symbols.get_serialized_ids(new_symbols)
    sentence.text = SymbolIdDict.symbols_to_str(new_symbols)
  return symbols, sentences


def sents_convert_to_ipa(sentences: SentenceList, text_symbols: SymbolIdDict, ignore_tones: bool, ignore_arcs: bool) -> Tuple[SymbolIdDict, SentenceList]:

  sents_new_symbols = []
  for sentence in sentences.items(True):
    new_symbols, new_accent_ids = symbols_convert_to_ipa(
      symbols=text_symbols.get_symbols(sentence.serialized_symbols),
      lang=sentence.lang,
      accent_ids=deserialize_list(sentence.serialized_accents),
      ignore_arcs=ignore_arcs,
      ignore_tones=ignore_tones
    )
    sentence.lang = Language.IPA
    sentence.serialized_accents = serialize_list(new_accent_ids)
    sents_new_symbols.append(new_symbols)

  return update_symbols_and_text(sentences, sents_new_symbols)


def sents_map(sentences: SentenceList, symbols_map: SymbolsMap) -> Tuple[SymbolIdDict, SentenceList]:
  sents_new_symbols = []
  result = SentenceList()
  counter = 0
  for sentence in sentences.items():
    symbols = text_to_symbols(sentence.text, sentence.lang)
    accent_ids = deserialize_list(sentence.serialized_accents)

    mapped_symbols = symbols_map.apply_to_symbols(symbols)

    text = SymbolIdDict.symbols_to_str(mapped_symbols)
    # a resulting empty text would make no problems
    sents = split_sentences(text, sentence.lang)
    for new_sent_text in sents:
      new_symbols = text_to_symbols(
        new_sent_text,
        lang=sentence.lang,
        ignore_tones=False,
        ignore_arcs=False
      )
      if len(accent_ids) > 0:
        new_accent_ids = [accent_ids[0]] * len(new_symbols)
      else:
        new_accent_ids = []

      tmp = Sentence(
        sent_id=counter,
        text=new_sent_text,
        lang=sentence.lang,
        serialized_accents=serialize_list(new_accent_ids),
        serialized_symbols=""
      )
      sents_new_symbols.append(new_symbols)

      result.append(tmp)
      counter += 1

  return update_symbols_and_text(sentences, sents_new_symbols)


# def sents_rules(sentences: SentenceList, rules: str) -> SentenceList:
#   pass


def sents_accent_template(sentences: SentenceList, text_symbols: SymbolIdDict, accent_ids: AccentsDict) -> AccentedSymbolList:
  res = AccentedSymbolList()
  for i, sent in enumerate(sentences.items()):
    symbols = text_symbols.get_symbols(sent.serialized_symbols)
    accents = accent_ids.get_accents(sent.serialized_accents)
    for j, symbol_accent in enumerate(zip(symbols, accents)):
      accented_symbol = AccentedSymbol(
        position=f"{i}-{j}",
        symbol=symbol_accent[0],
        accent=symbol_accent[1]
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
  return sentences


def prepare_for_inference(sentences: SentenceList, text_symbols: SymbolIdDict, known_symbols: SymbolIdDict) -> InferSentenceList:
  result = InferSentenceList()
  for sentence in sentences.items():
    old_text_symbols = text_symbols.get_symbols(sentence.serialized_symbols)
    infer_symbols = old_text_symbols
    
    if known_symbols.has_unknown_symbols(infer_symbols):
      infer_symbols = known_symbols.replace_unknown_symbols_with_pad(infer_symbols)

    infer_sentence = Sentence(
      sent_id=sentence.sent_id,
      lang=sentence.lang,
      serialized_accents=sentence.serialized_accents,
      serialized_symbols=known_symbols.get_serialized_ids(infer_symbols),
      text=SymbolIdDict.symbols_to_str(infer_symbols)
    )

    result.append(infer_sentence)
    # Maybe add info if something was unknown
  return result
