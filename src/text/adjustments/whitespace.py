import re

# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+')

def collapse_whitespace(text):
  return re.sub(_whitespace_re, ' ', text)

if __name__ == "__main__":
  abc = " aiehataih a ai    aei "
  print(collapse_whitespace(abc))