import os

from parser.DatasetParserBase import DatasetParserBase


def get_metadata_filepath(root_dir: str) -> str:
  result = os.path.join(root_dir, 'metadata.csv')
  return result

def get_wav_dirpath(root_dir) -> str:
  result = os.path.join(root_dir, 'wavs')
  return result


class LJSpeechDatasetParser(DatasetParserBase):
  def __init__(self, path: str):
    super().__init__(path)

    self.metadata_filepath = get_metadata_filepath(path)

    if not os.path.exists(self.metadata_filepath):
      print("Metadatafile not found:", self.metadata_filepath)
      raise Exception()

    self.wav_dirpath = get_wav_dirpath(path)

    if not os.path.exists(self.wav_dirpath):
      print("WAVs not found:", self.wav_dirpath)
      raise Exception()

  def _parse_core(self) -> tuple:
    index = 1
    result = []

    with open(self.metadata_filepath, encoding='utf-8') as f:
      for line in f:
        tmp = self._parse_line(line)
        result.append(tmp)

    return result

  def _parse_line(self, line: str) -> tuple:
    parts = line.strip().split('|')
    basename = parts[0]
    # parts[1] contains years, in parts[2] the years are written out
    # ex. ['LJ001-0045', '1469, 1470;', 'fourteen sixty-nine, fourteen seventy;']
    wav_path = os.path.join(self.wav_dirpath, '{}.wav'.format(basename))
    text = parts[2]
    tmp = (basename, text, wav_path)
    return tmp

if __name__ == "__main__":
  import sys
  sys.path.append('../../tacotron2')

  ensure_downloaded('/datasets/LJSpeech-1.1-tmp')
  #parser = LJSpeechDatasetParser('/datasets/LJSpeech-1.1-test')
  #result = parser.parse()
  #print(result)
