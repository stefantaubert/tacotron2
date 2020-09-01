from src.core.pre.text.pre import SymbolsDict, TextData, TextDataList
from src.core.pre.text.pre import convert_to_ipa as text_convert_to_ipa
from src.core.pre.text.pre import normalize as text_normalize
from src.core.pre.text.pre import preprocess as text_preprocess
#from src.core.pre.text.pipeline import process_input_text as text_to_symbols_pipeline

from src.core.pre.text.pre_inference import Sentence, SentenceList, add_text as infer_add, sents_normalize as infer_norm, sents_convert_to_ipa as infer_convert_ipa, sents_map as infer_map
