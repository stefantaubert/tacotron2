import os

from matplotlib import use as use_matplotlib_backend
use_matplotlib_backend("Agg")
from src.core.common import get_subdir

from src.core.pre import remove_silence_plot as remove_silence_plot_core, WavData
from src.core.common import plot_melspec, stack_images_vertically
from shutil import copyfile
import matplotlib.pylab as plt
from typing import List, Optional
import tempfile
from src.app.pre.wav import get_wav_dir, load_wav_csv
from src.app.pre.ds import get_ds_dir, load_ds_csv

def _save_orig_plot_if_not_exists(dest_dir: str, mel):
  path = os.path.join(dest_dir, "original.png")
  if not os.path.isfile(path):
    plot_melspec(mel, title="Original")
    plt.savefig(path, bbox_inches='tight')
  return path

def _save_orig_wav_if_not_exists(dest_dir: str, orig_path: str):
  path = os.path.join(dest_dir, "original.wav")
  if not os.path.isfile(path):
    copyfile(orig_path, path)

def _save_trimmed_plot_temp(dest_dir: str, mel):
  path = tempfile.mktemp(suffix=".png")
  plot_melspec(mel, title="Trimmed")
  plt.savefig(path, bbox_inches='tight')
  return path

def _save_comparison(dest_dir: str, dest_name: str, paths: List[str]) -> str:
  path = os.path.join(dest_dir, f"{dest_name}.png")
  stack_images_vertically(paths, path)
  return path

def _get_trim_root_dir(wav_dir: str):
  return get_subdir(wav_dir, "trim", create=True)

def _get_trim_dir(wav_dir: str, entry: WavData):
  return os.path.join(_get_trim_root_dir(wav_dir), str(entry.entry_id))

def remove_silence_plot(base_dir: str, ds_name: str, wav_name: str, chunk_size: int, threshold_start: float, threshold_end: float, buffer_start_ms: float, buffer_end_ms: float, entry_id: Optional[int] = None):
  ds_dir = get_ds_dir(base_dir, ds_name)
  wav_dir = get_wav_dir(ds_dir, wav_name)
  assert os.path.isdir(wav_dir)
  data = load_wav_csv(wav_dir)
  entry: WavData
  if entry_id == None:
    entry = data.get_random_entry()
  else:
    entry = data.get_entry(entry_id)
  
  dest_dir = _get_trim_dir(wav_dir, entry)
  os.makedirs(dest_dir, exist_ok=True)
  
  dest_name = f"cs={chunk_size},ts={threshold_start}dBFS,bs={buffer_start_ms}ms,te={threshold_end}dBFS,be={buffer_end_ms}ms"

  wav_trimmed = os.path.join(dest_dir, f"{dest_name}.wav")

  mel_orig, mel_trimmed = remove_silence_plot_core(
    wav_path=entry.wav,
    out_path=wav_trimmed,
    chunk_size=chunk_size,
    threshold_start=threshold_start,
    threshold_end=threshold_end,
    buffer_start_ms=buffer_start_ms,
    buffer_end_ms=buffer_end_ms
  )

  _save_orig_wav_if_not_exists(dest_dir, entry.wav)
  orig = _save_orig_plot_if_not_exists(dest_dir, mel_orig)
  trimmed = _save_trimmed_plot_temp(dest_dir, mel_trimmed)
  resulting_path = _save_comparison(dest_dir, dest_name, [orig, trimmed])
  os.remove(trimmed)

  print(f"Saved result to: {resulting_path}")

if __name__ == "__main__":
  remove_silence_plot(
    base_dir="/datasets/models/taco2pt_v3",
    ds_name="thchs",
    wav_name="16000kHz_normalized",
    threshold_start = -20,
    threshold_end = -30,
    chunk_size = 5,
    buffer_start_ms = 100,
    buffer_end_ms = 150
  )
