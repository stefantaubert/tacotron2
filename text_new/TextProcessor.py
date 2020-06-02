import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
from tqdm import tqdm

from src.CMUDict.CMUDict import CMUDict
from src.CMUDict.sentence_to_ipa import sentence_to_ipa
from src.etc.IPA_symbol_extraction import extract_symbols
from src.tac.hparams import hparams
from src.tac.preprocessing.parser.DatasetParserBase import DatasetParserBase
from src.tac.preprocessing.parser.UtteranceFormat import UtteranceFormat
from src.tac.preprocessing.text.adjustments.TextAdjuster import TextAdjuster
from src.tac.preprocessing.text.conversion.SymbolConverter import \
    get_from_symbols


def get_txt_dir(caching_dir: str) -> str:
  ''' The directory to write the preprocessed text into. '''
  return os.path.join(caching_dir, 'preprocessing/text')

def get_symbols_file(caching_dir: str) -> str:
  ''' The directory to write the symbolmappings into. '''
  return os.path.join(caching_dir, 'preprocessing/symbols.json')

class TextProcessor():
  def __init__(self, hp: hparams, caching_dir: str):
    self.hp = hp
    self.caching_dir = caching_dir
    self._set_paths()
    self._ensure_folders_exist()
    
    self.adjuster = TextAdjuster()

  def _set_paths(self):
    self.txt_dir = get_txt_dir(self.caching_dir)

  def _ensure_folders_exist(self):
    os.makedirs(self.caching_dir, exist_ok=True)
    os.makedirs(self.txt_dir, exist_ok=True)
  
  def _preprocess_utterances(self, dataset):
    utterances = dataset.parse()
    ds_format = dataset.get_format()
 
    if self.hp.convert_to_ipa:
      print("Initializing IPA dictionary...")
      self.cmudict = CMUDict()
      self.cmudict.load()
    
    print('preprocess text step 1...')
    processed_utterances = []
    all_symbols = set()
    # todo multicore
    for basename, text, _ in tqdm(utterances):
      symbols = []
      if ds_format == UtteranceFormat.ENG:
        adjusted_text = self.adjuster.adjust(text)

        if self.hp.convert_to_ipa:
            ipa = sentence_to_ipa(adjusted_text, self.cmudict)
            symbols = extract_symbols(ipa)
        else:
          symbols = list(adjusted_text)
      elif ds_format == UtteranceFormat.IPA:
        ipa = text
        symbols = extract_symbols(ipa)
      else:
        raise NotImplementedError()

      all_symbols.update(set(symbols))

      tmp = (basename, symbols)
      processed_utterances.append(tmp)

    return (processed_utterances, all_symbols)
  
  def _dump_all_symbols(self, converter):
    symbols_dump_path = get_symbols_file(self.caching_dir)
    converter.dump(symbols_dump_path)

  def _convert_to_sequence_and_save(self, processed_utterances, converter):
    result = []
    # todo multicore
    print('preprocess text step 2...')
    for basename, text in tqdm(processed_utterances):
      sequence = converter.text_to_sequence(text)

      # save to file
      txt_filename = '{}.npy'.format(basename)
      txt_path = os.path.join(self.txt_dir, txt_filename)
      np.save(txt_path, sequence, allow_pickle=False)

      text_length = len(sequence)
      tmp = (basename, text_length)
      result.append(tmp)

    return result

  def process(self, dataset: DatasetParserBase, n_jobs):
    print('parse dataset...')
    processed_utterances, all_symbols = self._preprocess_utterances(dataset)
    converter = get_from_symbols(all_symbols)
    self._dump_all_symbols(converter)
    result = self._convert_to_sequence_and_save(processed_utterances, converter)
    self.processing_result = result
    return result

  def show_stats(self):
    assert self.processing_result

    textlenght_sum = sum([int(m[1]) for m in self.processing_result])
    textlenght_max = max([int(m[1]) for m in self.processing_result])

    print('Written {} utterances'.format(len(self.processing_result)))
    print('Sum input length (text chars): {}'.format(textlenght_sum))
    print('Max input length (text chars): {}'.format(textlenght_max))

if __name__ == "__main__":
  from hparams import hparams
  from multiprocessing import cpu_count

  if __name__ == "__main__":
    from src.preprocessing.parser.LJSpeechDatasetParser import LJSpeechDatasetParser
    from src.preprocessing.parser.DummyIPADatasetParser import DummyIPADatasetParser
    
    #parser = LJSpeechDatasetParser('/datasets/LJSpeech-1.1-test')
    parser = DummyIPADatasetParser('/datasets/IPA-Dummy')

    processor = TextProcessor(
      hparams.parse(''),
      '/datasets/models/tacotron/cache'
    )
    
    processor.process(parser, cpu_count())
    processor.show_stats()
