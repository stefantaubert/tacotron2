import argparse
from utils import save_map_json
from text.symbol_converter import load_from_file
from collections import OrderedDict

def save(path: str, symbols: set) -> set:
  with open(path, 'w', encoding='utf-8') as f:
    res = '\n'.join(list(sorted(symbols)))
    res += '\nCount: ' + str(len(symbols))
    f.write(res)

def set_to_str(s: set):
  return "{} ({})".format(' '.join(sorted(s)), str(len(s)))

def comp(symA, symB, out, out_symbols):
  conv_a = load_from_file(symA)
  conv_b = load_from_file(symB)
  symbolsA = set(conv_a.get_symbols(include_subset_id=False, include_id=False))
  symbolsB = set(conv_b.get_symbols(include_subset_id=False, include_id=False))
  
  sym_mapping = OrderedDict()
  for b in symbolsB:
    if b in symbolsA:
      sym_mapping[b] = b
  
  for b in symbolsB:
    if b not in symbolsA:
      sym_mapping[b] = ""

  save_map_json(out, sym_mapping)
  with open(out_symbols, 'w', encoding='utf-8') as f:
    f.write('\n'.join(list(sorted(list(symbolsA)))))

  print("A:\n", set_to_str(symbolsA))
  print("B:\n", set_to_str(symbolsB))
  print("Only in A:\n", set_to_str(symbolsA.difference(symbolsB)))
  print("Only in B:\n", set_to_str(symbolsB.difference(symbolsA)))
  print("In A & B:\n", set_to_str(symbolsA.intersection(symbolsB)))
  print("A + B:\n", set_to_str(symbolsA.union(symbolsB)))

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--no_debugging', action='store_true')
  parser.add_argument('--a', type=str)
  parser.add_argument('--b', type=str)
  parser.add_argument('--out', type=str)
  parser.add_argument('--out_symbols', type=str)
  parser.add_argument('--reverse', action='store_true')

  args = parser.parse_args()

  if not args.no_debugging:
    args.a = "/datasets/models/symbols/ipa_en.json"
    args.b = "/datasets/models/symbols/ipa_chn.json"
    args.out = "/datasets/models/symbols/map.json"
    args.out_symbols = "/datasets/models/symbols/symbols.txt"
    args.reverse = True
  
  if args.reverse:
    comp(args.b, args.a, args.out, args.out_symbols)
  else:
    comp(args.a, args.b, args.out, args.out_symbols)
