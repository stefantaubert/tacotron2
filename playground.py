
import epitran


if __name__ == "__main__":

  epi = epitran.Epitran('eng-Latn')
  
  with open('examples/north.txt', 'r') as f:
    lines = f.readlines()
  text = '\n'.join(lines)
  print(epi.transliterate(text))
  print(epi.trans_delimiter(text, normpunc=True, ligatures=True))