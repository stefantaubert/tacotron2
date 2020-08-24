import datetime
import os
from shutil import copyfile

import matplotlib.pylab as plt
import numpy as np

from src.app.utils import add_console_out_to_logger, add_file_out_to_logger, init_logger
from src.app.io import (get_checkpoints_dir,
                     get_infer_log, get_inference_root_dir, save_infer_plot,
                     save_infer_wav)
from src.app.waveglow.io import get_train_dir, save_diff_plot, save_v
from src.core.common import (get_basename, get_custom_or_last_checkpoint,
                             get_parent_dirname, get_subdir, plot_melspec)
from src.core.waveglow import get_infer_logger
from src.core.waveglow import infer as infer_core


def get_infer_dir(train_dir: str, input_name: str, iteration: int):
  subdir_name = f"{datetime.datetime.now():%Y-%m-%d,%H-%M-%S},wav={input_name},it={iteration}"
  return get_subdir(get_inference_root_dir(train_dir), subdir_name, create=True)

def save_infer_orig_plot(infer_dir: str, mel: np.ndarray):
  plot_melspec(mel, title="Original")
  path = os.path.join(infer_dir, f"{get_parent_dirname(infer_dir)}_orig.png")
  plt.savefig(path, bbox_inches='tight')
  return path

def save_infer_orig_wav(infer_dir: str, wav_path_orig: str):
  path = os.path.join(infer_dir, f"{get_parent_dirname(infer_dir)}_orig.wav")
  copyfile(wav_path_orig, path)

def infer(base_dir: str, train_name: str, wav_path: str, custom_checkpoint: int = 0, sigma: float = 0.666, denoiser_strength: float = 0.01, sampling_rate: float = 22050):
  train_dir = get_train_dir(base_dir, train_name, create=False)
  assert os.path.isdir(train_dir)
  
  init_logger(get_infer_logger())
  input_name = get_basename(wav_path)
  checkpoint_path, iteration = get_custom_or_last_checkpoint(get_checkpoints_dir(train_dir), custom_checkpoint)
  infer_dir = get_infer_dir(train_dir, input_name, iteration)
  add_console_out_to_logger(get_infer_logger())
  add_file_out_to_logger(get_infer_logger(), get_infer_log(infer_dir))
  
  wav, wav_mel, orig_mel = infer_core(
    wav_path=wav_path,
    custom_hparams=custom_checkpoint,
    denoiser_strength=denoiser_strength,
    sigma=sigma,
    checkpoint_path=checkpoint_path
  )

  save_infer_wav(infer_dir, sampling_rate, wav)
  save_infer_plot(infer_dir, wav_mel)
  save_infer_orig_wav(infer_dir, wav_path)
  save_infer_orig_plot(infer_dir, orig_mel)
  score = save_diff_plot(infer_dir)
  save_v(infer_dir)

  get_infer_logger().info(f"Imagescore: {score*100}%")
  get_infer_logger().info(f"Saved output to: {infer_dir}")

if __name__ == "__main__":
  infer(
    base_dir="/datasets/models/taco2pt_v3",
    train_name="debug",
    wav_path="/datasets/LJSpeech-1.1-lite/wavs/LJ003-0347.wav"
  )
