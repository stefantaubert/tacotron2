from typing import List, Optional, Set
from src.core.common import Language, load_csv, save_csv, convert_to_ipa, text_to_symbols, split_sentences, normalize, get_unique_items
from dataclasses import dataclass
from src.core.common import SymbolIdDict, SymbolsMap, create_symbols_map

@dataclass()
class Sentence:
  sent_id: int
  text: str
  lang: Language
  # Contains only the symbols which are known
  infer_text: str
  infer_serialized_symbols: str

class SentenceList(List[Sentence]):
  def save(self, file_path: str):
    save_csv(self, file_path)

  @classmethod
  def load(cls, file_path: str):
    data = load_csv(file_path, Sentence)
    return cls(data)

  def get_occuring_symbols(self) -> Set[str]:
    return get_unique_items([text_to_symbols(x.text, x.lang) for x in self])

@dataclass()
class AccentedSymbol:
  position: int
  symbol: str
  accent_id: int

class AccentedSymbolList(List[AccentedSymbol]):
  def save(self, file_path: str):
    save_csv(self, file_path)

  @classmethod
  def load(cls, file_path: str):
    data = load_csv(file_path, AccentedSymbol)
    return cls(data)


def add_text(text: str, lang: Language, known_symbols: SymbolIdDict, accent_id: int = 0) -> SentenceList:
  res = SentenceList()
  sents = split_sentences(text, lang)
  for i, sent in enumerate(sents):
    symbols = text_to_symbols(sent, lang)
    symbol_ids = known_symbols.get_ids(symbols)
    #symbol_ids = known_symbols.symbols_to_ids(symbols, subset_id_if_multiple=accent_id, add_eos=False, replace_unknown_with_pad=True)
    infer_serialized_symbol_ids = SymbolIdDict.serialize_symbol_ids(symbol_ids)
    infer_text = known_symbols.get_text(symbol_ids)
    s = Sentence(i, sent, lang, infer_text, infer_serialized_symbol_ids)
    res.append(s)
  return res

def sents_normalize(sentences: SentenceList) -> SentenceList:
  for s in sentences:
    s.text = normalize(s.text, s.lang)
  return sentences

def sents_convert_to_ipa(sentences: SentenceList, ignore_tones: bool, ignore_arcs: bool, replace_unknown_by: str) -> SentenceList:

  for s in sentences:
    s.text = convert_to_ipa(s.text, s.lang)
    s.lang = Language.IPA

    symbols = text_to_symbols(s.text, s.lang, ignore_tones=ignore_tones, ignore_arcs=ignore_arcs, replace_unknown_ipa_by=replace_unknown_by)
    # Maybe add info if something was unknown
    # Theoretically arcs could be out but in rereading the text with extract_from_sentence they would be automatically included as far as i know through ipapy
    s.text = SymbolIdDict.symbols_to_str(symbols)
  return sentences

def sents_map(sentences: SentenceList, symbols_map: SymbolsMap) -> SentenceList:
  result = SentenceList()
  counter = 0
  for s in sentences:
    symbols = text_to_symbols(s.text, s.lang)
    mapped_symbols = symbols_map.apply_to_symbols(symbols)
    text = SymbolIdDict.symbols_to_str(mapped_symbols)
    # a resulting empty text would make no problems
    sents = split_sentences(text, s.lang)
    for new_sent_text in sents:
      tmp = Sentence(counter, new_sent_text, s.lang)
      result.append(tmp)
      counter += 1
  return result


def sents_rules(sentences: SentenceList, rules: str) -> SentenceList:
  pass

def sents_accent_template(sentences: SentenceList, rules: str) -> AccentedSymbolList:
  pass

def sents_accent_apply(sentences: SentenceList, accented_symbols: AccentedSymbolList) -> SentenceList:
  pass

if __name__ == "__main__":
  from src.core.common import read_text
  example_text = "This is a test. And an other one.\nAnd a new line.\r\nAnd a line with \r.\n\nAnd a line with \n in it. This is a question? This is a error!"
  #example_text = read_text("examples/en/democritus.txt")
  conv = SymbolIdDict.init_from_symbols({"T", "h", "i", "s"})
  sents = add_text(example_text, Language.ENG, conv)
  print(sents)
  sents = sents_normalize(sents)
  print(sents)
  #sents = sents_map(sents, symbols_map=SymbolsMap.from_tuples([("o", "b"), ("a", ".")]))
  print(sents)
  sents = sents_convert_to_ipa(sents, ignore_tones=True, ignore_arcs=True, replace_unknown_by="_")
  print(sents)
