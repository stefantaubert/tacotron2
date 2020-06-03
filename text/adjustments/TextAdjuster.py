'''
Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You'll typically want to use:
  1. "english_cleaners" for English text
  2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
     the Unidecode library (https://pypi.python.org/pypi/Unidecode)
  3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
     the symbols in symbols.py to match your data).
'''

from unidecode import unidecode as convert_to_ascii

from text.adjustments.abbreviations import \
    expand_abbreviations
from text.adjustments.numbers import normalize_numbers
from text.adjustments.whitespace import \
    collapse_whitespace


class TextAdjuster:
  def __init__(self):
    super().__init__()

  def adjust(self, text):
    '''Pipeline for English text, including number and abbreviation expansion.'''
    text = convert_to_ascii(text)
    # text = text.lower()
    # todo datetime conversion
    text = normalize_numbers(text)
    text = expand_abbreviations(text)
    text = collapse_whitespace(text)
    return text

if __name__ == "__main__":
  adj = TextAdjuster()
  inp = "hello my name is mr. test and    1 + 3 is 4.   "
  out = adj.adjust(inp)
  print(out)
