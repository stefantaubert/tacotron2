
dataset_path = '/datasets/LJSpeech-1.1'

from parser.LJSpeechDatasetParser import LJSpeechDatasetParser
from text.adjustments.TextAdjuster import TextAdjuster
from text.conversion.SymbolConverter import get_from_symbols
import pandas as pd
import os

p = LJSpeechDatasetParser(dataset_path)
p.parse()

#print(p.data)

adj = TextAdjuster()
data = []

### normalize input
for basename, text, wav_path in p.data:
  cleaned_text = adj.adjust(text)
  data.append((basename, cleaned_text, wav_path))

### get all symbols
symbols = set()
for _, text, _ in data:
  current_symbols = set(text)
  symbols = symbols.union(current_symbols)
conv = get_from_symbols(symbols)
conv.dump('/tmp/symbols.json')

### convert text to symbols
result = []
for basename, text, wav_path in data:
  current_symbols = conv.text_to_sequence(text)
  current_symbols_str = ",".join([str(s) for s in current_symbols])
  result.append((basename, wav_path, text, current_symbols_str))

### save
#dest_filename = os.path.join(dataset_path, 'preprocessed.txt')
dest_filename = "/tmp/preprocessed.csv"

pd.DataFrame(result).to_csv(dest_filename, header=None, index=None, sep="|")
