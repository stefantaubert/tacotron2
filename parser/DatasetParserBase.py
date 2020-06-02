import os

class DatasetParserBase():
  def __init__(self, path: str):
    super().__init__()

    self.data = None

    if not os.path.exists(path):
      print("Directory not found:", path)
      raise Exception()

    self.path = path
  
  def parse(self) -> tuple:
    ''' 
    returns tuples of each utterance string and wav filepath
    (basename, text, wav_path)
    '''
    print("reading utterances")

    data_is_already_parsed = self.data != None

    if not data_is_already_parsed:
      result = self._parse_core()
      self.data = result

    print("finished.")
    return self.data

  def _parse_core(self) -> tuple:
    pass
