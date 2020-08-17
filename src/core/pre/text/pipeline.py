from src.core.pre.text.ipa2symb import extract_from_sentence
import epitran
from src.core.common import Language
from nltk import download
from nltk.tokenize import sent_tokenize
from src.core.pre.text.symbol_converter import SymbolConverter
from src.core.pre.text.chn_tools import chn_to_ipa
from src.core.pre.text.adjustments import normalize_text
from typing import List, Tuple
import logging

def process_input_text(lines: List[str], ipa: bool, ignore_tones: bool, ignore_arcs: bool, subset_id: int, lang: Language, symbols_map: dict, conv: SymbolConverter, debug_logger: logging.Logger) -> List[List[int]]:
  if ipa:
    if lang == Language.ENG:
      epi = epitran.Epitran('eng-Latn')
    elif lang == Language.GER:
      epi = epitran.Epitran('deu-Latn')
  
  if lang == Language.ENG or lang == Language.GER:
    download('punkt', quiet=True)

  sentences = []
  for line in lines:
    if lang == Language.CHN or lang == Language.IPA:
      sents = line.split('.')
      sents = [x.strip() for x in sents]
      sents = [x + '.' for x in sents if x != '']
    elif lang == Language.ENG:
      sents = sent_tokenize(line, language="english")
    elif lang == Language.GER:
      sents = sent_tokenize(line, language="german")
    else:
      raise Exception("Unknown input language!")
    sentences.extend(sents)

  if lang == Language.IPA:
    accented_sents = sentences
  elif lang == Language.CHN:
    accented_sents = sentences
    if ipa:
      tmp = []
      for s in sentences:
        chn_ipa = chn_to_ipa(s, add_period=False)
        tmp.append(chn_ipa)
      accented_sents = tmp
  elif lang == Language.GER:
    accented_sents = sentences
    if ipa:
      tmp = []
      for s in sentences:
        ger_ipa = epi.transliterate(s)
        tmp.append(ger_ipa)
      accented_sents = tmp
  elif lang == Language.ENG:
    cleaned_sents = []
    for s in sentences:
      cleaned_sent = normalize_text(s)
      cleaned_sents.append(cleaned_sent)
    
    debug_logger.debug("Input normalized")
    debug_logger.debug(cleaned_sents)
   
    accented_sents = []
    for s in cleaned_sents:
      ### TODO include rules in next step under if block
      if ipa:
        accented_sentence = epi.transliterate(s)
      else:
        accented_sentence = s
      accented_sents.append(accented_sentence)

  debug_logger.debug("Accented sents")
  debug_logger.debug(accented_sents)
 
  if symbols_map:
    ipa_mapping = { k: extract_from_sentence(v, ignore_tones=ignore_tones, ignore_arcs=ignore_arcs) for k, v in symbols_map.items() }

  #print('\n'.join(sentences))
  seq_sents = []
  seq_sents_text = []
  seq_sents_ids = []
  unknown_symbols = set()
  for s in accented_sents:
    if ipa:
      symbols = extract_from_sentence(s, ignore_tones=ignore_tones, ignore_arcs=ignore_arcs)
    else:
      symbols = list(s)

    mapped_symbols = []
    if symbols_map:
      for sy in symbols:
        sy_is_mapped = sy in ipa_mapping.keys()
        if sy_is_mapped:
          if ipa_mapping[sy] == '':
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
    seq_sents_ids.append(symbol_ids)
    serialized_symbol_ids = SymbolConverter.serialize_symbol_ids(symbol_ids)
    seq_sents.append(f'{serialized_symbol_ids}\n')

  debug_logger.debug("Input symbols")
  debug_logger.debug(seq_sents)

  if symbols_map:
    debug_logger.debug("Input sentences mapped")
    debug_logger.debug(seq_sents_text)

  if len(unknown_symbols) > 0:
    debug_logger.info(f'Unknown symbols: {unknown_symbols}')
  else:
    debug_logger.info('There were no unknown symbols.')

  debug_logger.info("Text to synthesize processed.")

  return seq_sents_ids
