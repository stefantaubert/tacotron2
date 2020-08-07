from ipapy.ipastring import IPAString
from ipapy.ipachar import IPAChar, IPADiacritic
import re
import string
from typing import List

rx = '[{}]'.format(re.escape(string.punctuation))

arc = '͡'

def extract_from_sentence(ipa_sentence: str, ignore_tones: bool = False, ignore_arcs: bool = False):
  res = []
  tmp: List[str] = []

  for c in ipa_sentence:
    if c in string.punctuation or c in string.whitespace:
      if len(tmp) > 0:
        raw_word = ''.join(tmp)
        s_ipa = IPAString(unicode_string=raw_word, ignore=False)
        raw_word_symbols = extract_symbols(s_ipa, ignore_tones, ignore_arcs)
        res.extend(raw_word_symbols)
        tmp.clear()
      res.append(c)
    else:
      tmp.append(c)

  if len(tmp) > 0:
    raw_word = ''.join(tmp)
    s_ipa = IPAString(unicode_string=raw_word, ignore=False)
    raw_word_symbols = extract_symbols(s_ipa, ignore_tones, ignore_arcs)
    res.extend(raw_word_symbols)
    tmp.clear()
    
  return res


def extract_symbols(ipa: IPAString, ignore_tones: bool, ignore_arcs: bool) -> List[str]:
  symbols: List[str] = []

  for char in ipa.ipa_chars:
    if char.is_diacritic or char.is_tone:
      if len(symbols) > 0:
        if char.is_tone and ignore_tones:
          continue
        else:
          # I think it is a bug in IPAString that the arc sometimes gets classified as diacritic and sometimes not
          if char.unicode_repr == arc:
            if ignore_arcs:
              continue
            else:
              symbols.append(arc)
          else:
            symbols[-1] += char.unicode_repr
    else:
      uc = char.unicode_repr
      if ignore_arcs:
        uc = uc.split(arc)
        symbols.extend(uc)
      else:
        symbols.append(uc)

  return symbols

if __name__ == "__main__":
  y = u"p͡f"
  res = extract_from_sentence(y, ignore_tones=True, ignore_arcs=True)


  #y = u"ˈprɪnɪŋ, ɪn ðə ˈoʊnli sɛns wɪθ wɪʧ wi ər æt ˈprɛzənt kənˈsərnd, ˈdɪfərz frəm moʊst ɪf nɑt frəm ɔl ðə ɑrts ənd kræfts ˌrɛprɪˈzɛnɪd ɪn ðə ˌɛksəˈbɪʃən."
  #y = u"naw, æz ɔl bʊks nɑt pɹajmɛɹəli ɪntɛndəd æz pɪkt͡ʃɹ̩-bʊks kənsɪst pɹɪnsɪpli ʌv tajps kəmpowzd tə fɔɹm lɛtɹ̩pɹɛs"
  #y = u"ɪʧ kt͡ʃɹ̩?"
  #y = u"tɕy˥˩ɕi˥ meɪ˧˩˧kwɔ˧˥ tsʰan˥i˥˩ɥœn˥˩ i˧˩˧ tsʰɑʊ˧˩˧ni˧˩˧ i˥ fən˥˩ ʈʂɨ˥ʈʂʰɨ˧˥ kʰɤ˥˩lin˧˥twən˥˩ ɕjɑŋ˥˩ pwɔ˥xeɪ˥ pʰaɪ˥˩piŋ˥ tɤ tɕɥœ˧˥i˥˩an˥˩ ʈʂwən˧˩˧peɪ˥˩ tsaɪ˥˩ pən˧˩˧ɥœ˥˩ ʂɑŋ˥˩ɕyn˧˥ tɕin˥˩ɕiŋ˧˥ pjɑʊ˧˩˧tɕɥœ˧˥ t˥˩ʃ˥˩"
  #y = "t˥˩ʃ˥˩?"
  #y = u"wɪʧ"
  #y = "ɪʃn̩'"

  #res = extract_from_sentence(y, ignore_tones=False, ignore_arcs=False)
  #print(''.join(res))
  #res = extract_from_sentence(y, ignore_tones=True, ignore_arcs=False)
  #print(''.join(res))
  #res = extract_from_sentence(y, ignore_tones=False, ignore_arcs=True)
  # print(''.join(res))
  # print(res)
  # print(set(res))
  # print(len(set(res)))
  
