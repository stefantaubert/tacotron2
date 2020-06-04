if __name__ == "__main__":
  from text.adjustments import normalize_text
  from text.conversion.SymbolConverter import get_from_file
  import os
  from nltk.tokenize import sent_tokenize
  from script_ds_pre import symbols_path
  import epitran
  from ipa2symb import extract_from_sentence

  epi = epitran.Epitran('eng-Latn')
  conv = get_from_file(symbols_path)

  lines = []

  with open('in/text.txt', 'r') as f:
    lines = f.readlines()

  sentences = []
  for line in lines:
    sents = sent_tokenize(line)
    sentences.extend(sents)

  cleaned_sents = []
  for s in sentences:
    cleaned_sent = normalize_text(s)
    cleaned_sents.append(cleaned_sent)

  with open('in/text_sents.txt', 'w') as f:
    f.writelines(['{}\n'.format(s) for s in cleaned_sents])

  accented_sents = []
  for s in cleaned_sents:
    ipa_text = epi.transliterate(s)
    ### todo include rules
    accented_sentence = ipa_text
    accented_sents.append(accented_sentence)

  with open('in/text_sents_accented.txt', 'w') as f:
    f.writelines(['{}\n'.format(s) for s in accented_sents])

  #print('\n'.join(sentences))
  seq_sents = []
  for s in accented_sents:
    ipa_symbols = extract_from_sentence(s)
    s_seq = conv.text_to_sequence(ipa_symbols)
    s_seq_str = ','.join([str(x) for x in s_seq])
    seq_sents.append('{}\n'.format(s_seq_str))

  with open('in/text_sents_accented_seq.txt', 'w') as f:
    f.writelines(seq_sents)
    
  print("Text to synthesize processed.")