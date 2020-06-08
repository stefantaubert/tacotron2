import argparse
import os

import epitran
from nltk.tokenize import sent_tokenize

from ipa2symb import extract_from_sentence
from text.adjustments import normalize_text
from text.conversion.SymbolConverter import get_from_file
from paths import input_symbols, symbols_path, input_dir

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--base_dir', type=str, help='base directory', default='/datasets/models/taco2pt_ipa')
  parser.add_argument('--ipa', type=str, help='IPA-based', default='true')
  parser.add_argument('--text', type=str, help='path to text which should be synthesized', default='examples/stella.txt')
  parser.add_argument('--is_ipa', type=str, help='text is ipa', default='false')
  args = parser.parse_args()
  is_ipa = str.lower(args.is_ipa) == 'true'
  use_ipa = str.lower(args.ipa) == 'true' or is_ipa
  
  print("Processing text from", args.text)
  
  if use_ipa:
    epi = epitran.Epitran('eng-Latn')
  
  conv = get_from_file(os.path.join(args.base_dir, symbols_path))
  base=os.path.basename(args.text)
  file_name = os.path.splitext(base)[0]

  lines = []

  with open(args.text, 'r') as f:
    lines = f.readlines()

  sentences = []
  for line in lines:
    if is_ipa:
      sents = line.split('.')
      sents = [x.strip() for x in sents]
      sents = [x + '.' for x in sents if x != '']
    else:
      sents = sent_tokenize(line)
    sentences.extend(sents)

  if is_ipa:
    accented_sents = sentences
  else:
    cleaned_sents = []
    for s in sentences:
      cleaned_sent = normalize_text(s)
      cleaned_sents.append(cleaned_sent)

    with open(os.path.join(args.base_dir, input_dir, "normalized_sentences_{}.txt".format(file_name)), 'w') as f:
      f.writelines(['{}\n'.format(s) for s in cleaned_sents])

    accented_sents = []
    for s in cleaned_sents:
      ### TODO include rules in next step under if block
      if use_ipa:
        accented_sentence = epi.transliterate(s)
      else:
        accented_sentence = s
      accented_sents.append(accented_sentence)

  with open(os.path.join(args.base_dir, input_dir, "input_sentences_{}.txt".format(file_name)), 'w') as f:
    f.writelines(['{}\n'.format(s) for s in accented_sents])

  #print('\n'.join(sentences))
  seq_sents = []
  unknown_symbols = set()
  for s in accented_sents:
    if use_ipa:
      symbols = extract_from_sentence(s)
    else:
      symbols = list(s)
    unknown_symbols = unknown_symbols.union(conv.get_unknown_symbols(symbols))
    s_seq = conv.text_to_sequence(symbols)
    s_seq_str = ','.join([str(x) for x in s_seq])
    seq_sents.append('{}\n'.format(s_seq_str))

  with open(os.path.join(args.base_dir, input_symbols), 'w') as f:
    f.writelines(seq_sents)

  if len(unknown_symbols) > 0:
    print('Unknown symbols:', unknown_symbols)
  else:
    print('There were no unknown symbols.')

  print("Text to synthesize processed.")
