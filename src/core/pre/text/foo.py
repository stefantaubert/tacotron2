from typing import List, Optional
from src.core.common import Language
from nltk.tokenize import sent_tokenize
from src.core.pre.text.utils import split_chn_text, split_ipa_text
from src.core.pre.text.chn_tools import chn_to_ipa
from nltk import download
from src.core.common import load_csv, save_csv
from dataclasses import dataclass
import epitran
from src.core.pre.text.adjustments import normalize_text
from src.core.pre.text.ipa2symb import extract_from_sentence
from src.core.pre.text.symbol_converter import SymbolConverter
from src.core.pre.text.symbols_map import SymbolsMap, create_symbols_map

@dataclass()
class Sentence:
  sent_id: int
  text: str
  lang: Language

class SentenceList(List[Sentence]):
  def save(self, file_path: str):
    save_csv(self, file_path)

  @classmethod
  def load(cls, file_path: str):
    data = load_csv(file_path, Sentence)
    return cls(data)


def split_sentences(text: str, lang: Language) -> List[str]:
  if lang == Language.ENG or lang == Language.GER:
    download('punkt', quiet=True)
  
  if lang == Language.CHN:
    sentences = split_chn_text(text)
  elif lang == Language.IPA:
    sentences = split_ipa_text(text)
  elif lang == Language.ENG:
    sentences = sent_tokenize(text, language="english")
  elif lang == Language.GER:
    sentences = sent_tokenize(text, language="german")
  else:
    raise Exception("Unknown input language!")

  return sentences

def add_text(text: str, lang: Language) -> SentenceList:
  res = SentenceList()
  sents = split_sentences(text, lang)
  for i, sent in enumerate(sents):
    s = Sentence(i, sent, lang)
    res.append(s)
  return res

def sents_normalize(sentences: SentenceList) -> SentenceList:
  for s in sentences:
    if s.lang == Language.ENG:
      s.text = normalize_text(s.text)
    else:
      continue
  return sentences

def sents_convert_to_ipa(sentences: SentenceList, ignore_tones: bool, ignore_arcs: bool, replace_unknown_by: str) -> SentenceList:
  epi_en = epitran.Epitran('eng-Latn')
  epi_de = epitran.Epitran('deu-Latn')
  
  for s in sentences:
    if s.lang == Language.ENG:
      s.text = epi_en.transliterate(s.text)
    elif s.lang == Language.GER:
      s.text = epi_de.transliterate(s.text)
    elif s.lang == Language.CHN:
      s.text = chn_to_ipa(s.text)
    elif s.lang == Language.IPA:
      pass
    symbols = extract_from_sentence(s.text, ignore_tones=ignore_tones, ignore_arcs=ignore_arcs, replace_unknown_ipa_by=replace_unknown_by)
    # Maybe add info if something was unknown
    # Theoretically arcs could be out but in rereading the text with extract_from_sentence they would be automatically included as far as i know through ipapy
    s.text = SymbolConverter.symbols_to_str(symbols)
    s.lang = Language.IPA
  return sentences

def extract_symbols(sentence: str, lang: Language):
  if lang == Language.ENG:
    symbols = list(sentence)
  elif lang == Language.GER:
    symbols = list(sentence)
  elif lang == Language.CHN:
    symbols = list(sentence)
  elif lang == Language.IPA:
    symbols = extract_from_sentence(sentence, ignore_tones=False, ignore_arcs=False, replace_unknown_ipa_by="")
  return symbols
    
def sents_map(sentences: SentenceList, symbols_map: SymbolsMap) -> SentenceList:
  result = SentenceList()
  counter = 0
  for s in sentences:
    symbols = extract_symbols(s.text, s.lang)
    mapped_symbols = symbols_map.apply_to_symbols(symbols)
    text = SymbolConverter.symbols_to_str(mapped_symbols)
    # a resulting empty text would make no problems
    sents = split_sentences(text, s.lang)
    for new_sent_text in sents:
      tmp = Sentence(counter, new_sent_text, s.lang)
      result.append(tmp)
      counter += 1
  return result

if __name__ == "__main__":
  from src.core.common import read_text
  example_text = "This is a test. And an other one.\nAnd a new line.\r\nAnd a line with \r.\n\nAnd a line with \n in it. This is a question? This is a error!"
  example_text = read_text("examples/en/democritus.txt")
  sents = add_text(example_text, Language.ENG)
  print(sents)
  sents = sents_normalize(sents)
  print(sents)
  #sents = sents_map(sents, symbols_map=SymbolsMap.from_tuples([("o", "b"), ("a", ".")]))
  print(sents)
  sents = sents_convert_to_ipa(sents, ignore_tones=True, ignore_arcs=True, replace_unknown_by="_")
  print(sents)
