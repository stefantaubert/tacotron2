import argparse
import os

import epitran
from nltk.tokenize import sent_tokenize
from nltk import download

from src.text.ipa2symb import extract_from_sentence
from src.common.utils import parse_json
from src.text.adjustments import normalize_text
from src.text.symbol_converter import load_from_file, serialize_symbol_ids
from src.paths import get_symbols_path, inference_input_normalized_sentences_file_name, inference_input_sentences_file_name, inference_input_sentences_mapped_file_name, inference_input_symbols_file_name, inference_input_file_name, inference_input_map_file_name
from src.text.chn_tools import chn_to_ipa

def process_input_text(training_dir_path: str, infer_dir_path: str, ipa: bool, ignore_tones: bool, ignore_arcs: bool, subset_id: int, lang: str, use_map: bool):
  if ipa:
    if lang == "en":
      epi = epitran.Epitran('eng-Latn')
    elif lang == "ger":
      epi = epitran.Epitran('deu-Latn')

  conv = load_from_file(get_symbols_path(training_dir_path))
  
  lines = []

  input_file = os.path.join(infer_dir_path, inference_input_file_name)
  with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()
  
  is_ipa = lang == "ipa"
  if not is_ipa:
    download('punkt', quiet=True)

  sentences = []
  for line in lines:
    if lang == "chn" or lang == "ipa":
      sents = line.split('.')
      sents = [x.strip() for x in sents]
      sents = [x + '.' for x in sents if x != '']
    elif lang == "en":
      sents = sent_tokenize(line, language="english")
    elif lang == "ger":
      sents = sent_tokenize(line, language="german")
    else:
      raise Exception("Unknown input language!")
    sentences.extend(sents)

  if is_ipa:
    accented_sents = sentences
  elif lang == "chn":
    accented_sents = sentences
    if ipa:
      tmp = []
      for s in sentences:
        chn_ipa = chn_to_ipa(s, add_period=False)
        tmp.append(chn_ipa)
      accented_sents = tmp
  elif lang == "ger":
    accented_sents = sentences
    if ipa:
      tmp = []
      for s in sentences:
        chn_ipa = epi.transliterate(s)
        tmp.append(chn_ipa)
      accented_sents = tmp
  elif lang == "en":
    cleaned_sents = []
    for s in sentences:
      cleaned_sent = normalize_text(s)
      cleaned_sents.append(cleaned_sent)

    with open(os.path.join(infer_dir_path, inference_input_normalized_sentences_file_name), 'w', encoding='utf-8') as f:
      f.writelines(['{}\n'.format(s) for s in cleaned_sents])
   
    accented_sents = []
    for s in cleaned_sents:
      ### TODO include rules in next step under if block
      if ipa:
        accented_sentence = epi.transliterate(s)
      else:
        accented_sentence = s
      accented_sents.append(accented_sentence)

  with open(os.path.join(infer_dir_path, inference_input_sentences_file_name), 'w', encoding='utf-8') as f:
    f.writelines(['{}\n'.format(s) for s in accented_sents])

  if use_map:
    map_path = os.path.join(infer_dir_path, inference_input_map_file_name)
    ipa_mapping = parse_json(map_path)
    ipa_mapping = { k: extract_from_sentence(v, ignore_tones=ignore_tones, ignore_arcs=ignore_arcs) for k, v in ipa_mapping.items() }

  #print('\n'.join(sentences))
  seq_sents = []
  seq_sents_text = []
  unknown_symbols = set()
  for s in accented_sents:
    if ipa:
      symbols = extract_from_sentence(s, ignore_tones=ignore_tones, ignore_arcs=ignore_arcs)
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

  with open(os.path.join(infer_dir_path, inference_input_symbols_file_name), 'w', encoding='utf-8') as f:
    f.writelines(seq_sents)
  
  if use_map:
    with open(os.path.join(infer_dir_path, inference_input_sentences_mapped_file_name), 'w', encoding='utf-8') as f:
      f.writelines(['{}\n'.format(s) for s in seq_sents_text])

  if len(unknown_symbols) > 0:
    print('Unknown symbols:', unknown_symbols)
  else:
    print('There were no unknown symbols.')

  print("Text to synthesize processed.")
