import argparse
import os

import epitran
from nltk.tokenize import sent_tokenize

from ipa2symb import extract_from_sentence
from text.adjustments import normalize_text
from text.conversion.SymbolConverter import get_from_file
from paths import input_symbols, input_dir, symbols_path_name, filelist_dir

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--base_dir', type=str, help='base directory', default='/datasets/models/taco2pt_ms')
  parser.add_argument('--ipa', type=str, help='IPA-based', default='true')
  parser.add_argument('--text', type=str, help='path to text which should be synthesized', default='examples/north_chn.txt')
  parser.add_argument('--is_ipa', type=str, help='text is ipa', default='true')
  parser.add_argument('--ds_name', type=str, required=False, default='thchs', help='thchs or ljs')
  parser.add_argument('--speaker', type=str, required=False, default='A11', help='speaker')
  parser.add_argument('--map', default='')

  args = parser.parse_args()

  is_ipa = str.lower(args.is_ipa) == 'true'
  use_ipa = str.lower(args.ipa) == 'true' or is_ipa
  use_map = args.map != ''
  print("Processing text from", args.text)
  
  if use_ipa:
    epi = epitran.Epitran('eng-Latn')
  if use_map:
    print("Using mapping from:", args.map)
  else:
    print("Using no mapping.")

  speaker_dir = os.path.join(args.base_dir, filelist_dir, args.ds_name, args.speaker)
  conv = get_from_file(os.path.join(speaker_dir, symbols_path_name))
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

  if use_map:
    with open(args.map, 'r') as f:
      tmp = f.readlines()
    #ipa_mapping = {x.strip()[0]: x.strip()[-1] for x in tmp}
    ipa_mapping = { }
    for x in tmp:
      if '->' in x:
        from_to = x.rstrip('\n').replace(' -> ', '')
        symbs = extract_from_sentence(from_to)
        a = symbs[0]
        if len(symbs) > 2:
          b = symbs[1:]
        else:
          b = [symbs[1]]
      else:
        a = x.rstrip('\n')
        b = ''
      ipa_mapping[a] = b

  # for k, v in ipa_mapping.items():
  #   for sy in v:
  #     if not conv._is_valid_text_symbol(sy):
  #       print(k, '->', v, ',', sy, 'not in symbols')
  # with open(os.path.join(args.base_dir, input_dir, "input_sentences_mapped_{}.txt".format(file_name)), 'w') as f:
  #   f.writelines(['{}\n'.format(s) for s in res])

  #print('\n'.join(sentences))
  seq_sents = []
  seq_sents_text = []
  unknown_symbols = set()
  for s in accented_sents:
    if use_ipa:
      symbols = extract_from_sentence(s)
    else:
      symbols = list(s)

    mapped_symbols = []
    if use_map:
      for sy in symbols:
        sy_is_mapped = sy in ipa_mapping.keys()
        if sy_is_mapped:
          if ipa_mapping == '':
            continue
          else:
            mapped_symbols.extend(ipa_mapping[sy])
        else:
          mapped_symbols.append(sy)
    else:
      mapped_symbols = symbols

    unknown_symbols = unknown_symbols.union(conv.get_unknown_symbols(mapped_symbols))
    s_seq = conv.text_to_sequence(mapped_symbols)
    seq_sents_text.append(''.join(mapped_symbols))
    s_seq_str = ','.join([str(x) for x in s_seq])
    seq_sents.append('{}\n'.format(s_seq_str))

  with open(os.path.join(args.base_dir, input_symbols), 'w') as f:
    f.writelines(seq_sents)
  
  if use_map:
    with open(os.path.join(args.base_dir, input_dir, "input_sentences_mapped_{}.txt".format(file_name)), 'w') as f:
      f.writelines(['{}\n'.format(s) for s in seq_sents_text])

  if len(unknown_symbols) > 0:
    print('Unknown symbols:', unknown_symbols)
  else:
    print('There were no unknown symbols.')

  print("Text to synthesize processed.")
