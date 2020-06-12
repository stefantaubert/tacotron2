import numpy as np
from sklearn.manifold import TSNE
from nltk.corpus import stopwords
import plotly.offline as plt
import plotly.graph_objs as go
import json
from scipy.spatial import distance
import torch
from sklearn.preprocessing import normalize


models = [
  ('thchs_no_tone', '/datasets/models/taco2pt_ms/saved_checkpoints/thchs_no_tone_C17_ipa_2999', '/datasets/models/taco2pt_ms/filelist/thchs_no_tone/C17/symbols.json'),
  #('ljs_en', '/datasets/models/taco2pt_ms/saved_checkpoints/ljs_en_1_en_68500', '/datasets/models/taco2pt_ms/filelist/ljs_en/1/symbols.json'),
  ('ljs_ipa', '/datasets/models/taco2pt_ms/saved_checkpoints/ljs_1_ipa_49000', '/datasets/models/taco2pt_ms/filelist/ljs_ipa/1/symbols.json'),
  ('thchs', '/datasets/models/taco2pt_ms/saved_checkpoints/thchs_C17_ipa_2999', '/datasets/models/taco2pt_ms/filelist/thchs/C17/symbols.json'),
]

for name, model_path, symbols_path in models:
  with open(symbols_path, 'r', encoding='utf-8') as f:
    id_to_symbol = json.load(f)

  checkpoint_dict = torch.load(model_path, map_location='cpu')

  symbols = [(v,k) for k, v in id_to_symbol.items()]
  symbols.sort()
  symbols = [x[1] for x in symbols]
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
    res += "{}:\t{} ({:.2f}),\t{} ({:.2f}),\t{} ({:.2f})\n".format(symbol, s[0][1], s[0][0], s[1][1], s[1][0], s[2][1], s[2][0])
  with open(name + ".txt", 'w', encoding='utf-8') as f:
    f.write(res)

  # 3D
  tsne = TSNE(n_components=3, random_state=0)
  np.set_printoptions(suppress=True)
  Y = tsne.fit_transform(arr)
  x_coords = Y[:, 0]
  y_coords = Y[:, 1]
  z_coords = Y[:, 2]

  plot = [go.Scatter3d(x = x_coords,
                      y = y_coords,
                      z = z_coords,
                      mode = 'markers+text',
                      text = symbols,
                      textposition='bottom center',
                      hoverinfo = 'text',
                      marker=dict(size=5,opacity=0.8))]

  layout = go.Layout(title='Embeddings')
  fig = go.Figure(data=plot, layout=layout)
  plt.plot(fig, filename='{}_3d.html'.format(name))

  # 2D
  tsne = TSNE(n_components=2, random_state=0)
  np.set_printoptions(suppress=True)
  Y = tsne.fit_transform(arr)
  x_coords = Y[:, 0]
  y_coords = Y[:, 1]

  plot = [go.Scatter(x = x_coords,
                      y = y_coords,
                      mode = 'markers+text',
                      text = symbols,
                      textposition='bottom center',
                      hoverinfo = 'text',
                      marker=dict(size=5,opacity=0.8))]

  layout = go.Layout(title='Embeddings')
  fig = go.Figure(data=plot, layout=layout)
  plt.plot(fig, filename='{}_2d.html'.format(name))