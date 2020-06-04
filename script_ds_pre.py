symbols_path = 'filelist/symbols.json'
csv_separator = '\t'

if __name__ == "__main__":
  from parser.LJSpeechDatasetParser import LJSpeechDatasetParser
  from text.adjustments import normalize_text
  from text.conversion.SymbolConverter import get_from_symbols
  import pandas as pd
  import epitran
  from ipa2symb import extract_from_sentence
  from tqdm import tqdm
  import os

  epi = epitran.Epitran('eng-Latn')

  dataset_path = '/datasets/LJSpeech-1.1'

  p = LJSpeechDatasetParser(dataset_path)
  p.parse()

  #print(p.data)

  data = []

  ### normalize input
  for basename, text, wav_path in tqdm(p.data):
    normalized_text = normalize_text(text)
    ipa_text = epi.transliterate(normalized_text)
    ipa_symbols = extract_from_sentence(ipa_text)
    data.append((basename, normalized_text, ipa_text, ipa_symbols, wav_path))

  ### get all symbols
  symbols = set()
  for _, _, _, ipa, _ in data:
    current_symbols = set(ipa)
    print(current_symbols)
    symbols = symbols.union(current_symbols)
  conv = get_from_symbols(symbols)
  conv.dump(symbols_path)
  print(conv.get_symbols())

  ### convert text to symbols
  result = []
  for bn, norm_text, ipa_txt, ipa_sym, wav in data:
    seq = conv.text_to_sequence(ipa_sym)
    seq_str = ",".join([str(s) for s in seq])
    result.append((bn, wav_path, norm_text, ipa_txt, seq_str))

  ### save
  #dest_filename = os.path.join(dataset_path, 'preprocessed.txt')
  dest_filename = "/tmp/preprocessed.csv"

  df = pd.DataFrame(result)
  df1 = df.iloc[:, [1, 4]]
  df1.to_csv(dest_filename, header=None, index=None, sep=csv_separator)
  print("Dataset saved.")
  df2 = df.iloc[:, [0, 2, 3]]
  df2.to_csv(dest_filename + ".csv", header=None, index=None, sep=csv_separator)
  print("Dataset preprocessing finished.")
