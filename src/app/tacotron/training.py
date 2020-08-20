import os
import matplotlib
matplotlib.use('Agg')

from src.app.utils import add_console_out_to_logger, add_file_out_to_logger, reset_file_log, init_logger
from src.app.io import (get_checkpoints_dir,get_train_checkpoints_log_file,
                     get_train_log_file, get_train_logs_dir, load_trainset,
                     load_valset, save_testset, save_trainset,
                     save_valset)
from src.app.pre import (get_prepared_dir, load_filelist,
                         load_filelist_speakers_json,
                         load_filelist_symbol_converter)
from src.app.tacotron.io import get_train_dir
from src.core.pre import SpeakersIdDict, SymbolConverter, split_train_test_val
from src.core.tacotron import continue_train as continue_train_core
from src.core.tacotron import get_train_logger, get_checkpoints_eval_logger
from src.core.tacotron import train as train_core, load_symbol_embedding_weights_from, load_symbol_embedding_weights_from, get_uniform_weights, get_mapped_embedding_weights, SymbolsMap

_speakers_json = "speakers.json"
_symbols_json = "symbols.json"

def load_symbol_converter(train_dir: str) -> SymbolConverter:
  data_path = os.path.join(train_dir, _symbols_json)
  return SymbolConverter.load_from_file(data_path)
  
def save_symbol_converter(train_dir: str, data: SymbolConverter):
  data_path = os.path.join(train_dir, _symbols_json)
  data.dump(data_path)

def load_speakers_json(train_dir: str) -> SpeakersIdDict:
  speakers_path = os.path.join(train_dir, _speakers_json)
  return SpeakersIdDict.load(speakers_path)
  
def save_speakers_json(train_dir: str, speakers: SpeakersIdDict):
  speakers_path = os.path.join(train_dir, _speakers_json)
  speakers.save(speakers_path)

def train(base_dir: str, train_name: str, fl_name: str, warm_start_model: str = "", test_size: float = 0.01, validation_size: float = 0.05, hparams = "", split_seed: int = 1234, emb_map_model: str = "", emb_map_model_symbols: str = "", symbols_map_path: str = ""):
  prep_dir = get_prepared_dir(base_dir, fl_name)
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
  
  # todo log map
  mapped_emb_weights = None
  if emb_map_model:
    trained_weights = load_symbol_embedding_weights_from(emb_map_model)
    trained_symbols = SymbolConverter.load_from_file(emb_map_model_symbols)
    symbols_map = None
    if symbols_map_path:
      symbols_map = SymbolsMap.load(symbols_map_path)
    model_weights = get_uniform_weights(symbols_conv.get_symbol_ids_count(), trained_weights.shape[1])
    mapped_emb_weights = get_mapped_embedding_weights(
      model_weights=model_weights,
      model_symbols=symbols_conv,
      trained_weights=trained_weights,
      trained_symbols=trained_symbols,
      symbols_map=symbols_map,
    )

  train_core(
    warm_start_model_path=warm_start_model,
    mapped_emb_weights=mapped_emb_weights,
    custom_hparams=hparams,
    logdir=logs_dir,
    n_symbols=symbols_conv.get_symbol_ids_count(),
    n_speakers=len(speakers),
    trainset=trainset,
    valset=valset,
    save_checkpoint_dir=get_checkpoints_dir(train_dir)
  )

def continue_train(base_dir: str, train_name: str, hparams):
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
  mode = 3
  if mode == 1:
    train(
      base_dir="/datasets/models/taco2pt_v3",
      train_name="debug",
      fl_name="thchs",
      hparams="batch_size=17,iters_per_checkpoint=5,epochs_per_checkpoint=1,cache_mels=False"
    )
  elif mode == 2:
    model_path = "/datasets/models/taco2pt_v2/ljs_ipa_ms_from_scratch/checkpoints/113500"
    model_conv = "/datasets/models/taco2pt_v2/ljs_ipa_ms_from_scratch/filelist/symbols.json"
    train(
      base_dir="/datasets/models/taco2pt_v3",
      train_name="debug",
      fl_name="thchs",
      hparams="batch_size=17,iters_per_checkpoint=5,epochs_per_checkpoint=1,cache_mels=False",
      emb_map_model=model_path,
      emb_map_model_symbols=model_conv,
      symbols_map_path="maps/weights/chn_en_v1.json"
    )
  elif mode == 3:
    model_path = "/datasets/models/taco2pt_v2/ljs_ipa_ms_from_scratch/checkpoints/113500"
    model_conv = "/datasets/models/taco2pt_v2/ljs_ipa_ms_from_scratch/filelist/symbols.json"
    train(
      base_dir="/datasets/models/taco2pt_v3",
      train_name="debug",
      fl_name="thchs",
      hparams="batch_size=17,iters_per_checkpoint=5,epochs_per_checkpoint=1,cache_mels=False",
      emb_map_model=model_path,
      emb_map_model_symbols=model_conv
    )
  elif mode == 4:
    continue_train(
      base_dir="/datasets/models/taco2pt_v3",
      train_name="debug",
      hparams="batch_size=17,iters_per_checkpoint=100,epochs_per_checkpoint=1,cache_mels=True,use_saved_mels=True"
    )
