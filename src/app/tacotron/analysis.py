import os
from typing import Optional

import pandas as pd
import plotly.offline as plt

from src.app.io import get_checkpoints_dir
from src.app.tacotron.io import get_train_dir
from src.app.utils import prepare_logger
from src.core.common.analysis import plot_embeddings as plot_embeddings_core
from src.core.common.train import get_custom_or_last_checkpoint
from src.core.common.utils import get_subdir, save_df
from src.core.tacotron.training import CheckpointTacotron


def get_analysis_root_dir(train_dir: str):
  return get_subdir(train_dir, "analysis", create=True)


def _save_similarities_csv(analysis_dir: str, checkpoint_it: int, df: pd.DataFrame):
  path = os.path.join(analysis_dir, f"{checkpoint_it}.csv")
  save_df(df, path)


def _save_2d_plot(analysis_dir: str, checkpoint_it: int, fig):
  path = os.path.join(analysis_dir, f"{checkpoint_it}_2d.html")
  plt.plot(fig, filename=path, auto_open=False)


def _save_3d_plot(analysis_dir: str, checkpoint_it: int, fig):
  path = os.path.join(analysis_dir, f"{checkpoint_it}_3d.html")
  plt.plot(fig, filename=path, auto_open=False)


def plot_embeddings(base_dir: str, train_name: str, custom_checkpoint: Optional[int] = None):
  train_dir = get_train_dir(base_dir, train_name, create=False)
  assert os.path.isdir(train_dir)
  analysis_dir = get_analysis_root_dir(train_dir)

  logger = prepare_logger()

  checkpoint_path, checkpoint_it = get_custom_or_last_checkpoint(
    get_checkpoints_dir(train_dir), custom_checkpoint)
  checkpoint = CheckpointTacotron.load(checkpoint_path, logger)

  # pylint: disable=no-member
  text, fig_2d, fig_3d = plot_embeddings_core(
    symbols=checkpoint.get_symbols(),
    emb=checkpoint.get_symbol_embedding_weights(),
    logger=logger
  )

  _save_similarities_csv(analysis_dir, checkpoint_it, text)
  _save_2d_plot(analysis_dir, checkpoint_it, fig_2d)
  _save_3d_plot(analysis_dir, checkpoint_it, fig_3d)
  logger.info(f"Saved analysis to: {analysis_dir}")


if __name__ == "__main__":
  plot_embeddings(
    base_dir="/datasets/models/taco2pt_v5",
    train_name="debug"
  )
