import datetime
import os
from shutil import copyfile
from typing import List

import matplotlib.pylab as plt
import numpy as np
from tqdm import tqdm

from src.core.common import (Language, float_to_wav, get_basename,
                             get_custom_or_last_checkpoint, get_parent_dirname,
                             get_subdir, parse_json, plot_melspec,
                             stack_images_vertically)
from src.core.pre import PreparedDataList, SpeakersIdDict, SymbolConverter


def get_train_root_dir(base_dir: str, train_name: str, create: bool):
  return get_subdir(base_dir, os.path.join("train", train_name), create)

#region Training

def _get_train_csv(base_dir: str, train_name: str):
  return os.path.join(get_train_root_dir(base_dir, train_name, create=True), "train.csv")

def _get_test_csv(base_dir: str, train_name: str):
  return os.path.join(get_train_root_dir(base_dir, train_name, create=True), "test.csv")

def _get_val_csv(base_dir: str, train_name: str):
  return os.path.join(get_train_root_dir(base_dir, train_name, create=True), "validation.csv")

def _get_symbols_conv(base_dir: str, train_name: str):
  return os.path.join(get_train_root_dir(base_dir, train_name, create=True), "symbols.json")

def _get_speakers(base_dir: str, train_name: str):
  return os.path.join(get_train_root_dir(base_dir, train_name, create=True), "speakers.json")

def get_train_log_dir(base_dir: str, train_name: str):
  train_dir = get_train_root_dir(base_dir, train_name, create=True)
  return get_subdir(train_dir, "logs", create=True)

def get_train_log_file(base_dir: str, train_name: str):
  return os.path.join(get_train_log_dir(base_dir, train_name), "log.txt")

def get_checkpoints_dir(base_dir: str, train_name: str):
  train_dir = get_train_root_dir(base_dir, train_name, create=True)
  return get_subdir(train_dir, "checkpoints", create=True)

def save_trainset(base_dir: str, train_name: str, dataset: PreparedDataList):
  path = _get_train_csv(base_dir, train_name)
  dataset.save(path)

def load_trainset(base_dir: str, train_name: str) -> PreparedDataList:
  path = _get_train_csv(base_dir, train_name)
  return PreparedDataList.load(path)

def save_testset(base_dir: str, train_name: str, dataset: PreparedDataList):
  path = _get_test_csv(base_dir, train_name)
  dataset.save(path)

def load_testset(base_dir: str, train_name: str) -> PreparedDataList:
  path = _get_test_csv(base_dir, train_name)
  return PreparedDataList.load(path)
  
def save_valset(base_dir: str, train_name: str, dataset: PreparedDataList):
  path = _get_val_csv(base_dir, train_name)
  dataset.save(path)

def load_valset(base_dir: str, train_name: str) -> PreparedDataList:
  path = _get_val_csv(base_dir, train_name)
  return PreparedDataList.load(path)
  
def load_speakers_json(base_dir: str, train_name: str) -> SpeakersIdDict:
  speakers_path = _get_speakers(base_dir, train_name)
  return SpeakersIdDict.load(speakers_path)
  
def save_speakers_json(base_dir: str, train_name: str, speakers: SpeakersIdDict):
  speakers_path = _get_speakers(base_dir, train_name)
  speakers.save(speakers_path)

def load_symbol_converter(base_dir: str, train_name: str) -> SymbolConverter:
  data_path = _get_symbols_conv(base_dir, train_name)
  return SymbolConverter.load_from_file(data_path)
  
def save_symbol_converter(base_dir: str, train_name: str, data: SymbolConverter):
  data_path = _get_symbols_conv(base_dir, train_name)
  data.dump(data_path)

#endregion

#region Inference

def _get_inference_root_dir(base_dir: str, train_name: str):
  train_dir = get_train_root_dir(base_dir, train_name, create=True)
  return get_subdir(train_dir, "inference", create=True)

def load_infer_text(file_name: str) -> List[str]:
  with open(file_name, "r") as f:
    return f.readlines()

def load_infer_symbols_map(symbols_map: str) -> List[str]:
  return parse_json(symbols_map) 

def get_infer_dir(base_dir: str, train_name: str, input_name: str, iteration: int, speaker_id: int):
  subdir_name = "{}_text-{}_speaker-{}_it-{}".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), input_name, speaker_id, iteration)
  return get_subdir(_get_inference_root_dir(base_dir, train_name), subdir_name, create=True)

def get_infer_log(infer_dir: str):
  return os.path.join(infer_dir, f"{get_parent_dirname(infer_dir)}.txt")

def save_infer_wav(infer_dir: str, sampling_rate: int, wav: np.ndarray):
  path = os.path.join(infer_dir, f"{get_parent_dirname(infer_dir)}.wav")
  float_to_wav(wav, path, normalize=True, sample_rate=sampling_rate)

def save_infer_plot(infer_dir: str, mel: np.ndarray):
  plot_melspec(mel, title=get_parent_dirname(infer_dir))
  path = os.path.join(infer_dir, f"{get_parent_dirname(infer_dir)}.png")
  plt.savefig(path, bbox_inches='tight')

def save_infer_sentence_plot(infer_dir: str, sentence_nr: int, mel: np.ndarray):
  plot_melspec(mel, title="{}: {}".format(get_parent_dirname(infer_dir), sentence_nr))
  path = os.path.join(infer_dir, f"{sentence_nr}.png")
  plt.savefig(path, bbox_inches='tight')

def save_infer_pre_postnet_sentence_plot(infer_dir: str, sentence_nr: int, mel: np.ndarray):
  plot_melspec(mel, title="{}: Pre-Postnet {}".format(get_parent_dirname(infer_dir), sentence_nr))
  path = os.path.join(infer_dir, f"{sentence_nr}_pre_post.png")
  plt.savefig(path, bbox_inches='tight')

def save_infer_alignments_sentence_plot(infer_dir: str, sentence_nr: int, mel: np.ndarray):
  plot_melspec(mel, title="{}: Alignments {}".format(get_parent_dirname(infer_dir), sentence_nr))
  path = os.path.join(infer_dir, f"{sentence_nr}_alignments.png")
  plt.savefig(path, bbox_inches='tight')

def save_infer_wav_sentence(infer_dir: str, sentence_nr: int, sampling_rate: int, sent_wav: np.ndarray):
  path = os.path.join(infer_dir, f"{sentence_nr}.wav")
  float_to_wav(sent_wav, path, normalize=True, sample_rate=sampling_rate)

def save_infer_v_pre_post(infer_dir: str, sentence_ids: List[int]):
  paths = []
  for x in sentence_ids:
    paths.append(os.path.join(infer_dir, f"{x}_pre_post.png"))
  path = os.path.join(infer_dir, f"{get_parent_dirname(infer_dir)}_v_pre_post.png")
  stack_images_vertically(paths, path)

def save_infer_v_alignments(infer_dir: str, sentence_ids: List[int]):
  paths = []
  for x in sentence_ids:
    paths.append(os.path.join(infer_dir, f"{x}_alignments.png"))
  path = os.path.join(infer_dir, f"{get_parent_dirname(infer_dir)}_v_alignments.png")
  stack_images_vertically(paths, path)

def save_infer_v_plot(infer_dir: str, sentence_ids: List[int]):
  paths = []
  for x in sentence_ids:
    paths.append(os.path.join(infer_dir, f"{x}.png"))
  path = os.path.join(infer_dir, f"{get_parent_dirname(infer_dir)}_v.png")
  stack_images_vertically(paths, path)

#endregion

#region Validation

def _get_validation_root_dir(base_dir: str, train_name: str):
  train_dir = get_train_root_dir(base_dir, train_name, create=True)
  return get_subdir(train_dir, "validation", create=True)

def get_val_dir(base_dir: str, train_name: str, entry_id: int, iteration: int):
  subdir_name = "{}_id-{}_it-{}".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), entry_id, iteration)
  return get_subdir(_get_validation_root_dir(base_dir, train_name), subdir_name, create=True)

def save_val_plot(val_dir: str, mel):
  parent_dir = get_parent_dirname(val_dir)
  plot_melspec(mel, title=parent_dir)
  path = os.path.join(val_dir, f"{parent_dir}.png")
  plt.savefig(path, bbox_inches='tight')

def save_val_orig_plot(val_dir: str, mel):
  parent_dir = get_parent_dirname(val_dir)
  plot_melspec(mel, title=parent_dir)
  path = os.path.join(val_dir, f"{parent_dir}_orig.png")
  plt.savefig(path, bbox_inches='tight')

def save_val_comparison(val_dir: str):
  parent_dir = get_parent_dirname(val_dir)
  path1 = os.path.join(val_dir, f"{parent_dir}.png")
  path2 = os.path.join(val_dir, f"{parent_dir}_orig.png")
  assert os.path.exists(path1)
  assert os.path.exists(path2)
  path = os.path.join(val_dir, f"{parent_dir}_comp.png")
  stack_images_vertically([path1, path2], path)

def save_val_pre_postnet_plot(val_dir: str, mel):
  parent_dir = get_parent_dirname(val_dir)
  plot_melspec(mel, title=f"{parent_dir}: Pre-Postnet")
  path = os.path.join(val_dir, f"{parent_dir}_pre_post.png")
  plt.savefig(path, bbox_inches='tight')

def save_val_alignments_sentence_plot(val_dir: str, mel):
  parent_dir = get_parent_dirname(val_dir)
  plot_melspec(mel, title=f"{parent_dir}: Alignments")
  path = os.path.join(val_dir, f"{parent_dir}_alignments.png")
  plt.savefig(path, bbox_inches='tight')

def save_val_wav(val_dir: str, sampling_rate: int, wav) -> str:
  path = os.path.join(val_dir, f"{get_parent_dirname(val_dir)}.wav")
  float_to_wav(wav, path, normalize=True, sample_rate=sampling_rate)
  return path

def save_val_orig_wav(val_dir: str, wav_path: str):
  path = os.path.join(val_dir, f"{get_parent_dirname(val_dir)}_orig.wav")
  copyfile(wav_path, path)

def get_val_log(val_dir: str):
  return os.path.join(val_dir, f"{get_parent_dirname(val_dir)}.txt")

#endregion
