from ipapy.ipastring import IPAString
from ipapy.ipachar import IPAChar, IPADiacritic
import re
import string

rx = '[{}]'.format(re.escape(string.punctuation))

def extract_from_sentence(ipa_sentence: str):
  #words = ipa_sentence.split(' ')

  res = []
  tmp = []
  for c in ipa_sentence:
    if c in string.punctuation or c in string.whitespace:
      if len(tmp) > 0:
        raw_word = ''.join(tmp)
        s_ipa = IPAString(unicode_string=raw_word, ignore=False)
        raw_word_symbols = extract_symbols(s_ipa)
        res.extend(raw_word_symbols)
        tmp.clear()
      res.append(c)
    else:
      tmp.append(c)

  if len(tmp) > 0:
    raw_word = ''.join(tmp)
    s_ipa = IPAString(unicode_string=raw_word, ignore=False)
    raw_word_symbols = extract_symbols(s_ipa)
    res.extend(raw_word_symbols)
    tmp.clear()
    
  # for w in words:
  #   #replace_punctuation = string.maketrans(string.punctuation, ' ' * len(string.punctuation))
  #   #raw_word = w.maketrans(string.punctuation, ' ' * len(string.punctuation))
  #   #raw_word = w.strip(string.punctuation)
  #   #raw_word
  #   raw_word_parts = re.split(rx, w)
  #   for part in raw_word_parts:
  #     part_len = len(part)
  #     s_ipa = IPAString(unicode_string=part, ignore=False)
  #     raw_word_symbols = extract_symbols(s_ipa)
  #     #w.replace(raw_word, s_ipa)
  #     res.append(raw_word_symbols)
  #   #test = w.split(string.punctuation)
  #   #w.replace(raw_word, s_ipa)
  #   #res.append(raw_word_symbols)
  return res


def extract_symbols(ipa: IPAString):
  symbols = []

  for char in ipa.ipa_chars:
    if char.is_diacritic:
      if len(symbols) > 0:
        symbols[-1] += char.unicode_repr
    else:
      symbols.append(char.unicode_repr)

  return symbols

if __name__ == "__main__":
  y = u"ˈprɪnɪŋ, ɪn ðə ˈoʊnli sɛns wɪθ wɪʧ wi ər æt ˈprɛzənt kənˈsərnd, ˈdɪfərz frəm moʊst ɪf nɑt frəm ɔl ðə ɑrts ənd kræfts ˌrɛprɪˈzɛnɪd ɪn ðə ˌɛksəˈbɪʃən."
  y = u"naw, æz ɔl bʊks nɑt pɹajmɛɹəli ɪntɛndəd æz pɪkt͡ʃɹ̩-bʊks kənsɪst pɹɪnsɪpli ʌv tajps kəmpowzd tə fɔɹm lɛtɹ̩pɹɛs"
  #y = u"wɪʧ"
  #y = "ɪʃn̩'"
  res = extract_from_sentence(y)
  print(res)
  print(set(res))
  s_ipa = IPAString(unicode_string=y, ignore=True)
  for c in s_ipa.ipa_chars:
    print(c.unicode_repr)
  tmp = extract_symbols(y)
  print(tmp)
