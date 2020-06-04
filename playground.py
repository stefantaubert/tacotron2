import string
x = 'ænd ðə nɛkst jɪɹ ɡʌnθɹ̩ zajnɹ̩ æt ɔɡzbɹ̩ɡ fɑlowd sut; wajl ɪn fɔɹtin sɛvənti æt pɛɹɪs judælɹɪk ɡɪɹɪŋ ænd hɪz əsowsiɪts tɹ̩nd awt ðə fɹ̩st bʊks pɹɪntəd ɪn fɹæns, ɔlsow ɪn ɹowmən kɛɹɪktɹ̩.'
y = u"raw,pɪkt͡ʃɹ̩-bʊks,"

import re
rx = '[{}]'.format(re.escape(string.punctuation))
res = re.split(rx, y)
print(res)