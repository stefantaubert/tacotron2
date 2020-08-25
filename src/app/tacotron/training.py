import os
import matplotlib
matplotlib.use('Agg')
from typing import Optional

from src.app.utils import add_console_out_to_logger, add_file_out_to_logger, reset_file_log, init_logger
from src.app.io import (get_checkpoints_dir,get_train_checkpoints_log_file,
                      get_train_log_file, get_train_logs_dir, load_trainset,
                      load_valset, save_testset, save_trainset, save_speakers_json, load_speakers_json,
                      save_valset)
from src.app.pre import (get_prepared_dir, load_filelist,
                         load_filelist_speakers_json,
                         load_filelist_symbol_converter)
from src.app.tacotron.io import get_train_dir
from src.core.pre import SpeakersDict, SymbolConverter, split_train_test_val
from src.app.pre import load_weights_map
from src.core.tacotron import continue_train as continue_train_core
from src.core.tacotron import get_train_logger, get_checkpoints_eval_logger
from src.core.tacotron import train as train_core, load_symbol_embedding_weights_from, load_symbol_embedding_weights_from
from src.core.pre.text import SymbolsMap
import torch

_symbols_json = "symbols.json"

def load_symbol_converter(train_dir: str) -> SymbolConverter:
  data_path = os.path.join(train_dir, _symbols_json)
  return SymbolConverter.load_from_file(data_path)
  
def save_symbol_converter(train_dir: str, data: SymbolConverter):
  data_path = os.path.join(train_dir, _symbols_json)
  data.dump(data_path)

def _load_pretrained_weights(emb_map_model: str) -> Optional[torch.Tensor]:
  trained_weights = load_symbol_embedding_weights_from(emb_map_model) if emb_map_model else None
  return trained_weights

def _load_pretrained_symbol_conv(emb_map_model_symbols: str) -> Optional[SymbolConverter]:
  trained_symbols = SymbolConverter.load_from_file(emb_map_model_symbols) if emb_map_model_symbols else None
  return trained_symbols

def train(base_dir: str, train_name: str, prep_name: str, warm_start_model: Optional[str] = None, test_size: float = 0.01, validation_size: float = 0.05, hparams: Optional[str] = None, split_seed: int = 1234, emb_map_model: Optional[str] = None, emb_map_model_symbols: Optional[str] = None, symbols_map_path: Optional[str] = None):
  prep_dir = get_prepared_dir(base_dir, prep_name)
  wholeset = load_filelist(prep_dir)
  trainset, testset, valset = split_train_test_val(wholeset, test_size=test_size, val_size=validation_size, seed=split_seed, shuffle=True)
  train_dir = get_train_dir(base_dir, train_name, create=True)
  save_trainset(train_dir, trainset)
  save_testset(train_dir, testset)
  save_valset(train_dir, valset)

  symbols_conv = load_filelist_symbol_converter(prep_dir)
  save_symbol_converter(train_dir, symbols_conv)

  speakers = load_filelist_speakers_json(prep_dir)
  save_speakers_json(train_dir, speakers)

  init_logger(get_train_logger())
  init_logger(get_checkpoints_eval_logger())
  logs_dir = get_train_logs_dir(train_dir)
  log_file = get_train_log_file(logs_dir)
  checkpoints_log_file = get_train_checkpoints_log_file(logs_dir)
  reset_file_log(log_file)
  reset_file_log(checkpoints_log_file)
  add_console_out_to_logger(get_train_logger())
  add_console_out_to_logger(get_checkpoints_eval_logger())
  add_file_out_to_logger(get_train_logger(), log_file)
  add_file_out_to_logger(get_checkpoints_eval_logger(), checkpoints_log_file)
  
  train_core(
    warm_start_model_path=warm_start_model,
    custom_hparams=hparams,
    logdir=logs_dir,
    symbols_conv=symbols_conv,
    n_speakers=len(speakers),
    trainset=trainset,
    valset=valset,
    save_checkpoint_dir=get_checkpoints_dir(train_dir),
    trained_weights=_load_pretrained_weights(emb_map_model),
    symbols_map=load_weights_map(symbols_map_path),
    trained_symbols_conv=_load_pretrained_symbol_conv(emb_map_model_symbols)
  )

def continue_train(base_dir: str, train_name: str, hparams: Optional[str] = None):
  train_dir = get_train_dir(base_dir, train_name, create=False)
  assert os.path.isdir(train_dir)

  symbols_conv = load_symbol_converter(train_dir)
  speakers = load_speakers_json(train_dir)

  init_logger(get_train_logger())
  init_logger(get_checkpoints_eval_logger())
  logs_dir = get_train_logs_dir(train_dir)
  log_file = get_train_log_file(logs_dir)
  checkpoints_log_file = get_train_checkpoints_log_file(logs_dir)
  add_console_out_to_logger(get_train_logger())
  add_console_out_to_logger(get_checkpoints_eval_logger())
  add_file_out_to_logger(get_train_logger(), log_file)
  add_file_out_to_logger(get_checkpoints_eval_logger(), checkpoints_log_file)

  continue_train_core(
    custom_hparams=hparams,
    logdir=logs_dir,
    n_symbols=symbols_conv.get_symbol_ids_count(),
    n_speakers=len(speakers),
    trainset=load_trainset(train_dir),
    valset=load_valset(train_dir),
    save_checkpoint_dir=get_checkpoints_dir(train_dir)
  )


if __name__ == "__main__":
  mode = 2
  if mode == 1:
    train(
      base_dir="/datasets/models/taco2pt_v4",
      train_name="debug",
      prep_name="thchs",
      hparams="batch_size=17,iters_per_checkpoint=5,epochs_per_checkpoint=1,cache_mels=False"
    )
  elif mode == 2:
    model_path = "/datasets/models/taco2pt_v2/ljs_ipa_ms_from_scratch/checkpoints/113500"
    model_conv = "/datasets/models/taco2pt_v2/ljs_ipa_ms_from_scratch/filelist/symbols.json"
    train(
      base_dir="/datasets/models/taco2pt_v4",
      train_name="debug",
      prep_name="thchs",
      hparams="batch_size=17,iters_per_checkpoint=5,epochs_per_checkpoint=1,cache_mels=False",
      emb_map_model=model_path,
      emb_map_model_symbols=model_conv
    )
  elif mode == 3:
    model_path = "/datasets/models/taco2pt_v2/ljs_ipa_ms_from_scratch/checkpoints/113500"
    model_conv = "/datasets/models/taco2pt_v2/ljs_ipa_ms_from_scratch/filelist/symbols.json"
    train(
      base_dir="/datasets/models/taco2pt_v4",
      train_name="debug",
      prep_name="thchs",
      emb_map_model=model_path,
      emb_map_model_symbols=model_conv,
      symbols_map_path="maps/weights/chn_en_tones.json",
      warm_start_model=model_path,
      hparams="batch_size=17,iters_per_checkpoint=0,epochs_per_checkpoint=1,ignore_layers=[embedding.weight,speakers_embedding.weight],cache_mels=False"
    )
   
  elif mode == 4:
    continue_train(
      base_dir="/datasets/models/taco2pt_v4",
      train_name="debug",
      hparams="batch_size=17,iters_per_checkpoint=100,epochs_per_checkpoint=1,cache_mels=True,use_saved_mels=True"
    )
