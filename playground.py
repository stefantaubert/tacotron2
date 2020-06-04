import string
x = 'ænd ðə nɛkst jɪɹ ɡʌnθɹ̩ zajnɹ̩ æt ɔɡzbɹ̩ɡ fɑlowd sut; wajl ɪn fɔɹtin sɛvənti æt pɛɹɪs judælɹɪk ɡɪɹɪŋ ænd hɪz əsowsiɪts tɹ̩nd awt ðə fɹ̩st bʊks pɹɪntəd ɪn fɹæns, ɔlsow ɪn ɹowmən kɛɹɪktɹ̩.'
y = x
#y = u",raw,pɪkt͡ʃɹ̩-bʊk,s,"
#y = "raw"

import re
rx = '[{}]'.format(re.escape(string.punctuation))
res = re.split(rx, y)
print(res)

res = []
tmp = []
for c in y:
  if c in string.punctuation or c in string.whitespace:
    if len(tmp) > 0:
      raw_word = ''.join(tmp)
      res.append(raw_word)
      tmp.clear()
    res.append(c)
  else:
    tmp.append(c)

if len(tmp) > 0:
  raw_word = ''.join(tmp)
  res.append(raw_word)
  tmp.clear()

print(res)