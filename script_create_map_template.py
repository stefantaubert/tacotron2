import argparse
from utils import save_map_json, parse_map_json
from text.symbol_converter import load_from_file
from collections import OrderedDict
from ipa2symb import extract_from_sentence

def save(path: str, symbols: set) -> set:
  with open(path, 'w', encoding='utf-8') as f:
    res = '\n'.join(list(sorted(symbols)))
    res += '\nCount: ' + str(len(symbols))
    f.write(res)

def set_to_str(s: set):
  return "{} ({})".format(' '.join(sorted(s)), str(len(s)))

def read_symbols_from_text(corpora_path: str) -> set:
  with open(corpora_path, 'r', encoding='utf-8') as f:
    content = ''.join(f.readlines())
  content = content.replace('\n', '')
  symbols = set(extract_from_sentence(content, ignore_arcs=True, ignore_tones=True))
  return symbols

def read_symbols_from_file(path: str) -> set:
  conv = load_from_file(path)
  syms = set(conv.get_symbols(include_subset_id=False, include_id=False))
  return syms

def comp(symbolsA, symbolsB, out):
  only_a = list(sorted(list(symbolsA)))
  in_a_and_b = list(sorted(list(symbolsA.intersection(symbolsB))))
  only_in_b = list(sorted(list(symbolsB.difference(symbolsA))))
  
  sym_mapping = OrderedDict([(a, a) for a in in_a_and_b])

  for b in only_in_b:
    sym_mapping[b] = ""

  save_map_json(out, sym_mapping)
  symbols_out_file = "{}.symbols".format(out)
  with open(symbols_out_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(only_a))

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
  parser.add_argument('--existing_map', type=str, help="if your corpora extended and you want to extend an existing symbolsmap.")
  parser.add_argument('--mode', type=str, choices=["weights", "infer"])
  parser.add_argument('--out', type=str, default='/tmp/map.json')
  parser.add_argument('--reverse', action='store_true')

  args = parser.parse_args()

  if not args.no_debugging:
    args.mode = "weights"
    args.mode = "infer"
    if args.mode == "weights":
      args.a = "/datasets/models/symbols/ipa_en.json"
      args.b = "/datasets/models/symbols/ipa_chn.json"
      args.out = "/datasets/models/symbols/map.json"
      #args.reverse = True
    else:
      args.a = "/datasets/models/symbols/ipa_en.json"
      #args.a = "/datasets/models/symbols/ipa_chn.json"
      args.b = "examples/ipa/corpora.txt"
      args.out = "/datasets/models/symbols/en_v1.json"
      #args.existing_map = "/datasets/models/symbols/test.json"
  
  a = args.a
  b = args.b
  if args.mode == "weights":
    if args.reverse:
      a = args.b
      b = args.a
    syms_a = read_symbols_from_file(a)
    syms_b = read_symbols_from_file(b)
  elif args.mode == "infer":
    syms_a = read_symbols_from_file(a)
    syms_b = read_symbols_from_text(b)
    if args.existing_map:
      existing_map = parse_map_json(args.existing_map)
      existing_syms = set(existing_map.keys())
      print("Ignoring existing symbols: {}".format(set_to_str(existing_syms)))
      syms_b = syms_b.difference(existing_syms)
  comp(syms_a, syms_b, args.out)

