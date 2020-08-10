from typing import List, Tuple
from tqdm import tqdm
from src.core.pre.ds.data import DsDataList, DsData
from src.core.pre.text.data import TextData, TextDataList
from src.core.pre.language import Language
from src.text.adjustments import normalize_text
from src.text.symbol_converter import init_from_symbols, serialize_symbol_ids, SymbolConverter
from collections import OrderedDict, Counter

def process(data: DsDataList) -> Tuple[TextDataList, SymbolConverter, OrderedDict[str, int]]:
  result: TextDataList = []
  entry_symbols = []
  
  values: DsData
  for values in tqdm(data):
    symbols: List[str] = list(values.text)
    entry_symbols.append(symbols)
    result.append(TextData(values.entry_id, values.text, "", values.lang))

  symbol_counter = Counter(entry_symbols)
  all_symbols: OrderedDict[str, int] = OrderedDict(symbol_counter.most_common())
  conv: SymbolConverter = init_from_symbols(set(all_symbols.keys()))

  for i, symbols in enumerate(entry_symbols):
    symbol_ids = conv.symbols_to_ids(symbols, add_eos=True, replace_unknown_with_pad=True)
    result[i].serialized_symbol_ids = serialize_symbol_ids(symbol_ids)

  return result, conv, all_symbols
