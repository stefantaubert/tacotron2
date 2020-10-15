import re

# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+')


def collapse_whitespace(text: str) -> str:
  return re.sub(_whitespace_re, ' ', text)
