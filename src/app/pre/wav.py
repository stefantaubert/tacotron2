import os
from functools import partial
from typing import Any, Callable

from src.app.pre.ds import get_ds_dir, load_ds_csv
from src.app.utils import prepare_logger
from src.core.common.utils import get_subdir
from src.core.pre.wav import (WavData, WavDataList, log_stats, normalize,
                              preprocess, remove_silence, stereo_to_mono,
                              upsample)

_wav_data_csv = "data.csv"


def _get_wav_root_dir(ds_dir: str, create: bool = False):
  return get_subdir(ds_dir, "wav", create)


def get_wav_dir(ds_dir: str, wav_name: str, create: bool = False):
  return get_subdir(_get_wav_root_dir(ds_dir, create), wav_name, create)


def load_wav_csv(wav_dir: str) -> WavDataList:
  path = os.path.join(wav_dir, _wav_data_csv)
  return WavDataList.load(WavData, path)


def save_wav_csv(wav_dir: str, wav_data: WavDataList):
  os.makedirs(wav_dir, exist_ok=True)
  path = os.path.join(wav_dir, _wav_data_csv)
  wav_data.save(path)


def preprocess_wavs(base_dir: str, ds_name: str, wav_name: str):
  logger = prepare_logger()
  logger.info("Preprocessing wavs...")
  ds_dir = get_ds_dir(base_dir, ds_name)
  dest_wav_dir = get_wav_dir(ds_dir, wav_name)
  if os.path.isdir(dest_wav_dir):
    logger.error("Already exists.")
  else:
    data = load_ds_csv(ds_dir)
    wav_data = preprocess(data, dest_wav_dir, copy_wavs=True)
    save_wav_csv(dest_wav_dir, wav_data)
    ds_data = load_ds_csv(ds_dir)
    log_stats(ds_data, wav_data, logger)


def wavs_stats(base_dir: str, ds_name: str, wav_name: str):
  logger = prepare_logger()
  logger.info(f"Stats of {wav_name}")
  ds_dir = get_ds_dir(base_dir, ds_name)
  wav_dir = get_wav_dir(ds_dir, wav_name)
  if os.path.isdir(wav_dir):
    ds_data = load_ds_csv(ds_dir)
    wav_data = load_wav_csv(wav_dir)
    log_stats(ds_data, wav_data, logger)


def _wav_op(base_dir: str, ds_name: str, origin_wav_name: str, destination_wav_name: str, op: Callable[[WavDataList, str], WavDataList], logger):
  ds_dir = get_ds_dir(base_dir, ds_name)
  dest_wav_dir = get_wav_dir(ds_dir, destination_wav_name)
  if os.path.isdir(dest_wav_dir):
    logger.error("Already exists.")
  else:
    orig_wav_dir = get_wav_dir(ds_dir, origin_wav_name)
    assert os.path.isdir(orig_wav_dir)
    data = load_wav_csv(orig_wav_dir)
    wav_data = op(data, dest_wav_dir)
    save_wav_csv(dest_wav_dir, wav_data)
    ds_data = load_ds_csv(ds_dir)
    log_stats(ds_data, wav_data, logger)


def wavs_normalize(base_dir: str, ds_name: str, orig_wav_name: str, dest_wav_name: str):
  logger = prepare_logger()
  logger.info("Normalizing wavs...")
  op = partial(normalize)
  _wav_op(base_dir, ds_name, orig_wav_name, dest_wav_name, op, logger)


def wavs_upsample(base_dir: str, ds_name: str, orig_wav_name: str, dest_wav_name: str, rate: int):
  logger = prepare_logger()
  logger.info("Resampling wavs...")
  op = partial(upsample, new_rate=rate)
  _wav_op(base_dir, ds_name, orig_wav_name, dest_wav_name, op, logger)


def wavs_stereo_to_mono(base_dir: str, ds_name: str, orig_wav_name: str, dest_wav_name: str):
  logger = prepare_logger()
  logger.info("Converting wavs from stereo to mono...")
  op = partial(stereo_to_mono)
  _wav_op(base_dir, ds_name, orig_wav_name, dest_wav_name, op, logger)


def wavs_remove_silence(base_dir: str, ds_name: str, orig_wav_name: str, dest_wav_name: str, chunk_size: int, threshold_start: float, threshold_end: float, buffer_start_ms: float, buffer_end_ms: float):
  logger = prepare_logger()
  logger.info("Removing silence in wavs...")
  op = partial(remove_silence, chunk_size=chunk_size, threshold_start=threshold_start,
               threshold_end=threshold_end, buffer_start_ms=buffer_start_ms, buffer_end_ms=buffer_end_ms)
  _wav_op(base_dir, ds_name, orig_wav_name, dest_wav_name, op, logger)
