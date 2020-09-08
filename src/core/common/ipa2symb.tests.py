import unittest

import epitran

from src.core.common.ipa2symb import extract_from_sentence


class UnitTests(unittest.TestCase):

  def test_quick(self):
    y = u"p͡f"
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
    # print(''.join(res))
    #res = extract_from_sentence(y, ignore_tones=True, ignore_arcs=False)
    # print(''.join(res))
    #res = extract_from_sentence(y, ignore_tones=False, ignore_arcs=True)
    # print(''.join(res))
    # print(res)
    # print(set(res))
    # print(len(set(res)))a


if __name__ == '__main__':
  suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
  unittest.TextTestRunner(verbosity=2).run(suite)
