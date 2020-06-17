import argparse
import os

import epitran
from nltk.tokenize import sent_tokenize

from ipa2symb import extract_from_sentence
from utils import parse_map
from text.adjustments import normalize_text
from text.symbol_converter import load_from_file, serialize_symbol_ids
from paths import input_symbols, input_dir, symbols_path_name, filelist_dir

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--base_dir', type=str, help='base directory')
  parser.add_argument('--ipa', type=str, help='IPA-based')
  parser.add_argument('--text', type=str, help='path to text which should be synthesized')
  parser.add_argument('--is_ipa', type=str, help='text is ipa')
  parser.add_argument('--ds_name', type=str)
  parser.add_argument('--speaker', type=str)
  parser.add_argument('--map', default='')
  parser.add_argument('--subset_id', type=int)
  parser.add_argument('--debug', type=str, default='true')

  args = parser.parse_args()

  debug = str.lower(args.debug) == 'true'
  if debug:
    args.base_dir = '/datasets/models/taco2pt_ms'
    args.ipa = 'true'
    args.text = 'examples/grandfather.txt'
    args.is_ipa = 'false'
    if True:
      args.map = 'maps/en_chn.txt'
      args.subset_id = 1
    else:
      args.map = ''
      args.subset_id = 1
    speaker_dir = os.path.join(args.base_dir, filelist_dir)
  else:
    speaker_dir = os.path.join(args.base_dir, filelist_dir, args.ds_name, args.speaker)
  subset_id = args.subset_id
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

  conv = load_from_file(os.path.join(speaker_dir, symbols_path_name))
  base = os.path.basename(args.text)
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
    ipa_mapping = parse_map(args.map)
  
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
    seq_sents_text.append(''.join(mapped_symbols))
    if subset_id != None:
      symbol_ids = conv.symbols_to_ids(mapped_symbols, add_eos=True, replace_unknown_with_pad=True, subset_id_if_multiple=subset_id) #TODO: experiment if pad yes no
    else:  
      symbol_ids = conv.symbols_to_ids(mapped_symbols, add_eos=True, replace_unknown_with_pad=True) #TODO: experiment if pad yes no
    serialized_symbol_ids = serialize_symbol_ids(symbol_ids)
    seq_sents.append('{}\n'.format(serialized_symbol_ids))

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
