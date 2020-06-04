import argparse
import os

import epitran
from nltk.tokenize import sent_tokenize

from ipa2symb import extract_from_sentence
from text.adjustments import normalize_text
from text.conversion.SymbolConverter import get_from_file
from paths import input_text, input_text_sents, input_text_sents_accented, input_symbols, symbols_path

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-b', '--base_dir', type=str, help='base directory', default='/datasets/models/taco2pt')
  args = parser.parse_args()
  
  epi = epitran.Epitran('eng-Latn')
  conv = get_from_file(os.path.join(args.base_dir, symbols_path))

  lines = []

  with open(os.path.join(args.base_dir, input_text), 'r') as f:
    lines = f.readlines()

  sentences = []
  for line in lines:
    sents = sent_tokenize(line)
    sentences.extend(sents)

  cleaned_sents = []
  for s in sentences:
    cleaned_sent = normalize_text(s)
    cleaned_sents.append(cleaned_sent)

  with open(os.path.join(args.base_dir, input_text_sents), 'w') as f:
    f.writelines(['{}\n'.format(s) for s in cleaned_sents])

  accented_sents = []
  for s in cleaned_sents:
    ipa_text = epi.transliterate(s)
    ### todo include rules
    accented_sentence = ipa_text
    accented_sents.append(accented_sentence)

  with open(os.path.join(args.base_dir, input_text_sents_accented), 'w') as f:
    f.writelines(['{}\n'.format(s) for s in accented_sents])

  #print('\n'.join(sentences))
  seq_sents = []
  for s in accented_sents:
    ipa_symbols = extract_from_sentence(s)
    s_seq = conv.text_to_sequence(ipa_symbols)
    s_seq_str = ','.join([str(x) for x in s_seq])
    seq_sents.append('{}\n'.format(s_seq_str))

  with open(os.path.join(args.base_dir, input_symbols), 'w') as f:
    f.writelines(seq_sents)

  print("Text to synthesize processed.")
