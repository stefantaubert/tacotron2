from ipapy.ipastring import IPAString
from ipapy.ipachar import IPAChar, IPADiacritic
import re
import string
from typing import List

_rx = '[{}]'.format(re.escape(string.punctuation))

_arc = '͡'

def extract_from_sentence(ipa_sentence: str, ignore_tones: bool = False, ignore_arcs: bool = False, replace_unknown_ipa_by: str = '_'):
  res = []
  tmp: List[str] = []

  for c in ipa_sentence:
    if c in string.punctuation or c in string.whitespace:
      if len(tmp) > 0:
        raw_word_symbols = _extract_symbols(tmp, ignore_tones, ignore_arcs, replace_unknown_ipa_by)
        res.extend(raw_word_symbols)
        tmp.clear()
      res.append(c)
    else:
      tmp.append(c)

  if len(tmp):
    raw_word_symbols = _extract_symbols(tmp, ignore_tones, ignore_arcs, replace_unknown_ipa_by)
    res.extend(raw_word_symbols)
    tmp.clear()
  return res

def _extract_symbols(input_symbols: List[str], ignore_tones: bool, ignore_arcs: bool, replace_unknown_ipa_by: str) -> List[str]:
  symbols: List[str] = []
  input_word = ''.join(input_symbols)
  try:
    ipa = IPAString(unicode_string=input_word, ignore=False)
  except:
    ipa = IPAString(unicode_string=input_word, ignore=True)
    print(f"{input_word} conversion to IPA failed. Result would be: {ipa}.")
    result = [replace_unknown_ipa_by] * len(input_symbols)
    return result

  for char in ipa.ipa_chars:
    if char.is_diacritic or char.is_tone:
      if len(symbols) > 0:
        if char.is_tone and ignore_tones:
          continue
        else:
          # I think it is a bug in IPAString that the arc sometimes gets classified as diacritic and sometimes not
          if char.unicode_repr == _arc:
            if ignore_arcs:
              continue
            else:
              symbols.append(_arc)
          else:
            symbols[-1] += char.unicode_repr
    else:
      uc = char.unicode_repr
      if ignore_arcs:
        uc = uc.split(_arc)
        symbols.extend(uc)
      else:
        symbols.append(uc)

  return symbols

if __name__ == "__main__":
  y = u"p͡f"
  import epitran
  epi = epitran.Epitran('eng-Latn')
  y = epi.transliterate("At Müller's execution there was great competition for front seats,")
  #y += " ɡɹât͡ʃi"
  y += "？"
  res = extract_from_sentence(y, ignore_tones=True, ignore_arcs=True)
  print(res)

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
  
