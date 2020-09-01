from dragonmapper import hanzi
import re

_mappings = [
  ("。", "."),
  ("？", "?"),
  ("！", "!"),
  ("，", ","),
  ("：", ":"),
  ("；", ";"),
  ("「", "\""),
  ("」", "\""),
  ("『", "\""),
  ("』", "\""),
  ("、", ",")
]

_subs = [(re.compile(f'\{x[0]}'), x[1]) for x in _mappings]

def chn_to_ipa(chn: str):
  chn_words = chn.split(' ')
  res = []
  for w in chn_words:
    chn_ipa = hanzi.to_ipa(w)
    chn_ipa = chn_ipa.replace(' ', '')    
    res.append(chn_ipa)
  res_str = ' '.join(res)
  for regex, replacement in _subs:
    res_str = re.sub(regex, replacement, res_str)

  return res_str

if __name__ == "__main__":
  w = "东北军 的 一些 爱？」"
  res = chn_to_ipa(w)
  print(res)