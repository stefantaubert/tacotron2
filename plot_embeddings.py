import numpy as np
from sklearn.manifold import TSNE
from nltk.corpus import stopwords
import plotly.offline as plt
import plotly.graph_objs as go
import json
from scipy.spatial import distance
import torch
from sklearn.preprocessing import normalize
from text.symbol_converter import load_from_file
from paths import get_symbols_path, get_analysis_dir, analysis_2d_file_name, analysis_3d_file_name, analysis_sims_file_name, get_checkpoint_dir
import os
import argparse
from train import get_last_checkpoint

def analyse(training_dir_path: str, custom_checkpoint: int):
  conv = load_from_file(get_symbols_path(training_dir_path))
  symbols = conv.get_symbols(include_subset_id=False, include_id=False)

  if custom_checkpoint:
    checkpoint = custom_checkpoint
  else:
    checkpoint = get_last_checkpoint(training_dir_path)
  print("Analyzing checkpoint {}...".format(str(checkpoint)))
  checkpoint_path = os.path.join(get_checkpoint_dir(training_dir_path), checkpoint)
  checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')

  #symbols = [(v,k) for k, v in id_to_symbol.items()]
  #xsymbols.sort()
  #symbols = [x[1] for x in symbols]
  arr = np.empty((0, 512), dtype='f')
  emb = checkpoint_dict['state_dict']['embedding.weight']
  print("Emb len", len(emb))
  print("Sym len", len(symbols))
  assert len(emb) == len(symbols)
  for i, symbol in enumerate(symbols):
    wrd_vector = emb[i].numpy()
    norm2 = normalize(wrd_vector[:,np.newaxis], axis=0).ravel()
    arr = np.append(arr, np.array([norm2]), axis=0)

  sims = {}
  for i, symbol in enumerate(symbols):
    vec = arr[i]
    tmp = []
    for j, other_symbol in enumerate(symbols):
      if i != j:
        other_vec = arr[j]
        dist = distance.euclidean(vec, other_vec)
        if dist != float("-inf") and dist != float("inf") and other_symbol != "_" and other_symbol != "~":
          tmp.append((dist, other_symbol))
        else:
          tmp.append((float("inf"), other_symbol))
    tmp.sort()
    sims[symbol] = tmp
  #print(sims)
  res = ''
  for symbol, s in sims.items():
    res += "{}\t->\t{} ({:.2f})\t->\t{} ({:.2f})\t->\t{} ({:.2f})\n".format(symbol, s[0][1], s[0][0], s[1][1], s[1][0], s[2][1], s[2][0])
  dest_txt = os.path.join(get_analysis_dir(training_dir_path), "{}_{}".format(str(checkpoint), analysis_sims_file_name))
  with open(dest_txt, 'w', encoding='utf-8') as f:
    f.write(res)

  # 3D
  tsne = TSNE(n_components=3, random_state=0)
  np.set_printoptions(suppress=True)
  Y = tsne.fit_transform(arr)
  x_coords = Y[:, 0]
  y_coords = Y[:, 1]
  z_coords = Y[:, 2]

  plot = [go.Scatter3d(x = x_coords, y = y_coords, z = z_coords, mode = 'markers+text', text = symbols, textposition='bottom center', hoverinfo = 'text', marker=dict(size=5,opacity=0.8))]

  layout = go.Layout(title='Embeddings')
  fig = go.Figure(data=plot, layout=layout)
  plt.plot(fig, filename=os.path.join(get_analysis_dir(training_dir_path), "{}_{}".format(str(checkpoint), analysis_3d_file_name)))

  # 2D
  tsne = TSNE(n_components=2, random_state=0)
  np.set_printoptions(suppress=True)
  Y = tsne.fit_transform(arr)
  x_coords = Y[:, 0]
  y_coords = Y[:, 1]

  plot = [go.Scatter(x = x_coords, y = y_coords, mode = 'markers+text', text = symbols, textposition='bottom center', hoverinfo = 'text', marker=dict(size=5,opacity=0.8))]

  layout = go.Layout(title='Embeddings')
  fig = go.Figure(data=plot, layout=layout)
  plt.plot(fig, filename=os.path.join(get_analysis_dir(training_dir_path), "{}_{}".format(str(checkpoint), analysis_2d_file_name)))

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--no_debugging', action='store_true')
  parser.add_argument('--base_dir', type=str, help='base directory')
  parser.add_argument('--training_dir', type=str)
  parser.add_argument('--custom_checkpoint', type=str)

  args = parser.parse_args()
  
  if not args.no_debugging:
    args.base_dir = '/datasets/models/taco2pt_v2'
    args.training_dir = 'debug_ljs_ms'
    #args.custom_checkpoint = 0

  training_dir_path = os.path.join(args.base_dir, args.training_dir)
  analyse(training_dir_path, args.custom_checkpoint)

#   models = [
#     ('ljs_ipa_thchs_no_tone_A11', os.path.join(args.base_dir, savecheckpoints_dir, 'ljs_ipa_thchs_no_tone_A11_1499'), os.path.join(args.base_dir, savecheckpoints_dir, 'ljs_ipa_thchs_no_tone_A11_1499.json')),
#     ('ljs_en', os.path.join(args.base_dir, savecheckpoints_dir, 'ljs_en_1_ipa_51500'), os.path.join(args.base_dir, filelist_dir, 'ljs_en/1/symbols.json')),
#     ('thchs_no_tone', os.path.join(args.base_dir, savecheckpoints_dir, 'thchs_no_tone_C17_ipa_2999'), os.path.join(args.base_dir, filelist_dir, 'thchs_no_tone/C17/symbols.json')),
#     ('ljs_ipa', os.path.join(args.base_dir, savecheckpoints_dir, 'ljs_1_ipa_49000'), os.path.join(args.base_dir, filelist_dir, 'ljs_ipa/1/symbols.json')),
#     ('thchs', os.path.join(args.base_dir, savecheckpoints_dir, 'thchs_C17_ipa_2999'), os.path.join(args.base_dir, filelist_dir, 'thchs/C17/symbols.json')),
#   ]

#   analyse(models, args.base_dir, include_plotting=False)
