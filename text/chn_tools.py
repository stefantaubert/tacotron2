from dragonmapper import hanzi

__question_particle1 = '吗'
__question_particle2 = '呢'

def chn_to_ipa(chn, add_period: bool = False):
  chn_words = chn.split(' ')
  res = []
  for w in chn_words:
    is_question = str.endswith(w, __question_particle1) or str.endswith(w, __question_particle2)
    chn_ipa = hanzi.to_ipa(w)
    chn_ipa = chn_ipa.replace(' ', '')
    if is_question:
      chn_ipa += '?'
    res.append(chn_ipa)
  res = ' '.join(res)
  if res != '' and not str.endswith(res, '?') and add_period:
    res += '.'
  return res
