from unidecode import unidecode as convert_to_ascii

from text.adjustments.abbreviations import expand_abbreviations
from text.adjustments.numbers import normalize_numbers
from text.adjustments.whitespace import collapse_whitespace


def normalize_text(text):
  '''Pipeline for English text, including number and abbreviation expansion.'''
  text = convert_to_ascii(text)
  # text = text.lower()
  ### todo datetime conversion, BC to beecee
  text = normalize_numbers(text)
  text = expand_abbreviations(text)
  text = collapse_whitespace(text)
  return text

if __name__ == "__main__":
  inp = "hello my name is mr. test and    1 + 3 is 4.   "
  out = normalize_text(inp)
  print(out)
