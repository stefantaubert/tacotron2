from src.core.common import save_json, parse_json
from src.text.symbol_converter import load_from_file
from collections import OrderedDict
from src.core.pre import extract_from_sentence

def save(path: str, symbols: set) -> set:
  with open(path, 'w', encoding='utf-8') as f:
    res = '\n'.join(list(sorted(symbols)))
    res += '\nCount: ' + str(len(symbols))
    f.write(res)

def set_to_str(s: set):
  return "{} ({})".format(' '.join(sorted(s)), str(len(s)))

def read_symbols_from_text(corpora_path: str, ignore_tones: bool, ignore_arcs: bool) -> set:
  with open(corpora_path, 'r', encoding='utf-8') as f:
    content = ''.join(f.readlines())
  content = content.replace('\n', '')
  symbols = set(extract_from_sentence(content, ignore_arcs=ignore_arcs, ignore_tones=ignore_tones))
  return symbols

def read_symbols_from_file(path: str) -> set:
  parsed = parse_json(path)
  syms = set(parsed.keys())
  return syms

def comp(symbolsA: set, symbolsB: set, out: str):
  only_a = list(sorted(list(symbolsA)))
  in_a_and_b = list(sorted(list(symbolsA.intersection(symbolsB))))
  only_in_b = list(sorted(list(symbolsB.difference(symbolsA))))
  
  sym_mapping = OrderedDict([(a, a) for a in in_a_and_b])

  for b in only_in_b:
    sym_mapping[b] = ""

  save_json(out, sym_mapping)
  symbols_out_file = "{}.symbols".format(out)
  with open(symbols_out_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(only_a))

  print("A:\n", set_to_str(symbolsA))
  print("B:\n", set_to_str(symbolsB))
  print("Only in A:\n", set_to_str(symbolsA.difference(symbolsB)))
  print("Only in B:\n", set_to_str(symbolsB.difference(symbolsA)))
  print("In A & B:\n", set_to_str(symbolsA.intersection(symbolsB)))
  print("A + B:\n", set_to_str(symbolsA.union(symbolsB)))


def init_create_map_parser(parser):
  parser.add_argument('--a', type=str, required=True)
  parser.add_argument('--b', type=str, required=True)
  parser.add_argument('--existing_map', type=str, help="if your corpora extended and you want to extend an existing symbolsmap.")
  parser.add_argument('--mode', type=str, choices=["weights", "infer"], required=True)
  parser.add_argument('--out', type=str, default='/tmp/map.json')
  parser.add_argument('--reverse', action='store_true')
  parser.add_argument('--ignore_tones', action='store_true')
  parser.add_argument('--ignore_arcs', action='store_true')
  return __main

def __main(a, b, existing_map, mode, out, reverse, ignore_tones, ignore_arcs):

  skip = False
  if mode == "weights":
    if reverse:
      a = b
      b = a
    syms_a = read_symbols_from_file(a)
    syms_b = read_symbols_from_file(b)
  elif mode == "infer":
    syms_a = read_symbols_from_file(a)
    syms_b = read_symbols_from_text(b, ignore_tones, ignore_arcs)
    if existing_map:
      existing_map = parse_json(existing_map)
      existing_syms = set(existing_map.keys())
      print("Ignoring existing symbols: {}".format(set_to_str(existing_syms)))
      syms_b = syms_b.difference(existing_syms)
      if len(syms_b) == 0:
        print("There are no new symbols in the corpora.")
        skip = True
    
  if not skip:
    comp(syms_a, syms_b, out)


if __name__ == "__main__":
  mode = "weights"
  mode = "infer"
  if mode == "weights":
    a = "/datasets/models/taco2pt_v2/ds/ljs_ipa_v2/all_symbols.json"
    b = "/datasets/models/taco2pt_v2/ds/thchs_nosil_tones/all_symbols.json"
    #out = "/datasets/models/symbols/map.json"
    #reverse = True
    ignore_tones = False
    ignore_arcs = False
  else:
    a = "/datasets/models/taco2pt_v2/ds/ljs_ipa_v2/all_symbols.json"
    #a = "/datasets/models/taco2pt_v2/ds/thchs_v5/all_symbols.json"
    b = "examples/ipa/corpora.txt"
    #out = "/datasets/models/symbols/en_v1.json"
    existing_map = "maps/inference/en_v1.json"
    ignore_tones = True
    ignore_arcs = True
    
  __main(
    a=a,
    b=b,
    existing_map=existing_map,
    mode=mode,
    out="/tmp/out.json",
    reverse=False,
    ignore_tones=ignore_tones,
    ignore_arcs=ignore_arcs
  )
  