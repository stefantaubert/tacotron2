from typing import List
import re

def _split_text(text: str, separators: List[str]) -> List[str]:
  pattern = "|".join(separators)
  sents = re.split(f'({pattern})', text)
  res = []
  for i in range(len(sents)):
    if i % 2 == 0:
      res.append(sents[i])
      if i + 1 < len(sents):
        res[-1] += sents[i+1]
  res = [x.strip() for x in res]
  res = [x for x in res if x]
  return res

_question_mark = "？"
_exklamation_mark = "！"
_full_stop = "。"

def split_ipa_text(text: str) -> List[str]:
  separators = ["?", "!", "."]
  return _split_text(text, separators)

def split_chn_text(text: str) -> List[str]:
  separators = ["？", "！", "。"]
  return _split_text(text, separators)
 
