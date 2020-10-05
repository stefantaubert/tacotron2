from logging import Logger
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import torch
from scipy.spatial.distance import cosine
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

from src.core.common.symbol_id_dict import SymbolIdDict


def norm2emb(emb: torch.Tensor) -> np.ndarray:
  tmp = []
  for i in range(emb.shape[0]):
    word_vector = emb[i].numpy()
    norm2 = normalize(word_vector[:, np.newaxis], axis=0).ravel()
    tmp.append(np.array([norm2]))
  res = np.concatenate(tmp, axis=0)
  return res


def get_similarities(emb: np.ndarray) -> Dict[int, List[Tuple[int, float]]]:
  sims = {}
  for i in range(len(emb)):
    vec = emb[i]
    tmp = []
    for j in range(len(emb)):
      if i != j:
        other_vec = emb[j]
        dist = 1 - cosine(vec, other_vec)
        tmp.append((j, dist))
    tmp.sort(key=lambda x: x[1], reverse=True)
    sims[i] = tmp
  return sims


def sims_to_csv(sims: Dict[int, List[Tuple[int, float]]], symbols: SymbolIdDict) -> pd.DataFrame:
  lines = []
  assert len(sims) == len(symbols)
  for symbol_id, similarities in sims.items():
    sims = [f"{symbols.get_symbol(symbol_id)}", "<=>"]
    for other_symbol_id, similarity in similarities:
      sims.append(symbols.get_symbol(other_symbol_id))
      sims.append(f"{similarity:.2f}")
    lines.append(sims)
  df = pd.DataFrame(lines)
  return df


def emb_plot_3d(emb: np.ndarray, symbols: SymbolIdDict) -> go.Figure:
  np.set_printoptions(suppress=True)

  tsne = TSNE(n_components=3, random_state=0)
  Y = tsne.fit_transform(emb)
  x_coords = Y[:, 0]
  y_coords = Y[:, 1]
  z_coords = Y[:, 2]

  plot_3d = go.Scatter3d(x=x_coords, y=y_coords, z=z_coords, mode='markers+text', text=symbols,
                         textposition='bottom center', hoverinfo='text', marker=dict(size=5, opacity=0.8))

  layout = go.Layout(title='3D-Embeddings')
  fig = go.Figure(data=[plot_3d], layout=layout)
  return fig


def emb_plot_2d(emb: np.ndarray, symbols: SymbolIdDict) -> go.Figure:
  np.set_printoptions(suppress=True)
  tsne_2d = TSNE(n_components=2, random_state=0)
  Y_2d = tsne_2d.fit_transform(emb)
  x_coords = Y_2d[:, 0]
  y_coords = Y_2d[:, 1]
  plot_2d = go.Scatter(x=x_coords, y=y_coords, mode='markers+text', text=symbols,
                       textposition='bottom center', hoverinfo='text', marker=dict(size=5, opacity=0.8))

  layout = go.Layout(title='2D-Embeddings')
  fig = go.Figure(data=[plot_2d], layout=layout)
  return fig


def plot_embeddings(symbols: SymbolIdDict, emb: torch.Tensor, logger: Logger) -> Tuple[pd.DataFrame, go.Figure, go.Figure]:
  assert emb.shape[0] == len(symbols)

  logger.info(f"Emb size {emb.shape}")
  logger.info(f"Sym len {len(symbols)}")

  sims = get_similarities(emb.numpy())
  df = sims_to_csv(sims, symbols)
  all_symbols = symbols.get_all_symbols()
  emb_normed = norm2emb(emb)
  fig_2d = emb_plot_2d(emb_normed, all_symbols)
  fig_3d = emb_plot_3d(emb_normed, all_symbols)

  return df, fig_2d, fig_3d
