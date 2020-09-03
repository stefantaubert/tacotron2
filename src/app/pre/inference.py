import os

from src.core.common import get_subdir, read_text, Language
from src.core.pre import Sentence, SentenceList, infer_add, infer_convert_ipa, infer_map, infer_norm
from src.app.pre.prepare import get_prepared_dir

_text_csv = "text.csv"

def _get_text_dir(prep_dir: str, text_name: str):
  return get_subdir(prep_dir, text_name, create=True)

def load_text_csv(text_dir: str, text_name: str) -> SentenceList:
  path = os.path.join(text_dir, _text_csv)
  return SentenceList.load(Sentence, path)

def save_text_csv(text_dir: str, data: SentenceList):
  path = os.path.join(text_dir, _text_csv)
  data.save(path)

def add_text(base_dir: str, prep_name: str, text_name: str, filepath: str, lang: Language):
  prep_dir = get_prepared_dir(base_dir, prep_name, create=False)
  if not os.path.isdir(prep_dir):
    print("Please prepare data first.")
  else:
    text = read_text(filepath)
    data = infer_add(text, lang)
    text_dir = _get_text_dir(prep_dir, text_name)
    save_text_csv(text_dir, data)

if __name__ == "__main__":
  add_text(
    base_dir="/datasets/models/taco2pt_v5",
    prep_name="ljs_ipa",
    text_name="north",
    filepath="examples/en/north.txt",
    lang=Language.ENG
  )
