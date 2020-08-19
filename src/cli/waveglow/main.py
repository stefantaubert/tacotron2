import logging
import os
from argparse import ArgumentParser

from tqdm import tqdm

from src.cli import add_console_and_file_out_to_logger, reset_log
from src.cli.pre import (load_filelist, load_filelist_speakers_json,
                         load_filelist_symbol_converter)
from src.cli.waveglow.io import (get_checkpoints_dir, get_infer_dir,
                                 get_infer_log, get_train_log_dir,
                                 get_train_log_file, get_train_root_dir,
                                 get_val_dir, get_val_log, load_trainset,
                                 load_valset, save_diff_plot,
                                 save_infer_orig_plot, save_infer_orig_wav,
                                 save_infer_plot, save_infer_wav, save_testset,
                                 save_trainset, save_v, save_val_orig_plot,
                                 save_val_orig_wav, save_val_plot,
                                 save_val_wav, save_valset)
from src.core.common import (Language, get_basename,
                             get_custom_or_last_checkpoint)
from src.core.pre import (PreparedDataList, SpeakersIdDict, SymbolConverter,
                          split_train_test_val)
from src.core.waveglow import get_infer_logger, get_train_logger
from src.core.waveglow import infer as infer_core
from src.core.waveglow import train as train_core
from src.core.waveglow import validate as validate_core


def train(base_dir: str, train_name: str, fl_name: str, test_size: float = 0.01, validation_size: float = 0.01, hparams = "", split_seed: int = 1234):
  wholeset = load_filelist(base_dir, fl_name)
  trainset, testset, valset = split_train_test_val(wholeset, test_size=test_size, val_size=validation_size, seed=split_seed, shuffle=True)
  save_trainset(base_dir, train_name, trainset)
  save_testset(base_dir, train_name, testset)
  save_valset(base_dir, train_name, valset)

  log_file = get_train_log_file(base_dir, train_name)
  reset_log(log_file)
  add_console_and_file_out_to_logger(get_train_logger(), log_file)
  # todo log map & args

  train_core(
    custom_hparams=hparams,
    logdir=get_train_log_dir(base_dir, train_name),
    trainset=trainset,
    valset=valset,
    save_checkpoint_dir=get_checkpoints_dir(base_dir, train_name),
    continue_train=False
  )

def continue_train(base_dir: str, train_name: str, hparams):
  assert os.path.isdir(get_train_root_dir(base_dir, train_name, create=False))
  add_console_and_file_out_to_logger(get_train_logger(), get_train_log_file(base_dir, train_name))

  train_core(
    custom_hparams=hparams,
    logdir=get_train_log_dir(base_dir, train_name),
    trainset=load_trainset(base_dir, train_name),
    valset=load_valset(base_dir, train_name),
    save_checkpoint_dir=get_checkpoints_dir(base_dir, train_name),
    continue_train=True
  )

def validate(base_dir: str, train_name: str, entry_id: int, custom_checkpoint: int = 0, sigma: float = 0.666, denoiser_strength: float = 0.01, sampling_rate: float = 22050):
  logger = get_infer_logger()
  val = load_valset(base_dir, train_name)
  entry = val.get_entry(entry_id)
  checkpoint_path, iteration = get_custom_or_last_checkpoint(get_checkpoints_dir(base_dir, train_name), custom_checkpoint)
  val_dir = get_val_dir(base_dir, train_name, entry_id, iteration)
  add_console_and_file_out_to_logger(logger, get_val_log(val_dir))

  wav, wav_mel, orig_mel = validate_core(
    entry=entry,
    custom_hparams=custom_checkpoint,
    denoiser_strength=denoiser_strength,
    sigma=sigma,
    checkpoint_path=checkpoint_path
  )

  save_val_wav(val_dir, sampling_rate, wav)
  save_val_plot(val_dir, wav_mel)
  save_val_orig_wav(val_dir, entry.wav_path)
  save_val_orig_plot(val_dir, orig_mel)
  save_diff_plot(val_dir)
  save_v(val_dir)
  
  logger.info(f"Saved output to: {val_dir}")


def infer(base_dir: str, train_name: str, wav_path: str, custom_checkpoint: int = 0, sigma: float = 0.666, denoiser_strength: float = 0.01, sampling_rate: float = 22050):
  logger = get_infer_logger()
  input_name = get_basename(wav_path)
  checkpoint_path, iteration = get_custom_or_last_checkpoint(get_checkpoints_dir(base_dir, train_name), custom_checkpoint)
  infer_dir = get_infer_dir(base_dir, train_name, input_name, iteration)
  add_console_and_file_out_to_logger(logger, get_infer_log(infer_dir))
  
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
  save_diff_plot(infer_dir)
  save_v(infer_dir)
  
  logger.info(f"Saved output to: {infer_dir}")

if __name__ == "__main__":
  mode = 3
  if mode == 1:
    train(
      base_dir="/datasets/models/taco2pt_v2",
      train_name="debug_wg",
      fl_name="thchs",
      hparams="batch_size=4,iters_per_checkpoint=50,cache_wavs=False"
    )
  elif mode == 2:
    continue_train(
      base_dir="/datasets/models/taco2pt_v2",
      train_name="debug_wg",
      hparams="batch_size=4,iters_per_checkpoint=50,cache_wavs=False"
    )
  elif mode == 3:
    validate(
      base_dir="/datasets/models/taco2pt_v2",
      train_name="debug_wg",
      entry_id=31,
    )
  elif mode == 4:
    infer(
      base_dir="/datasets/models/taco2pt_v2",
      train_name="debug_wg",
      wav_path="/datasets/LJSpeech-1.1-lite/wavs/LJ003-0347.wav"
    )
