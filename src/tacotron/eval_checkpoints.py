import os
import time
import argparse
import math
from numpy import finfo
from tqdm import tqdm
import numpy as np

import torch
#from distributed_tacotron import apply_gradient_allreduce
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import argparse
from torch.utils.data import DataLoader

from src.tacotron.model import Tacotron2
from src.tacotron.data_utils import SymbolsMelLoader, SymbolsMelCollate
from src.tacotron.loss_function import Tacotron2Loss
from src.tacotron.logger import Tacotron2Logger
from src.tacotron.hparams import create_hparams
from src.common.utils import parse_ds_speakers, get_total_duration_min_df

from src.text.symbol_converter import load_from_file
from src.paths import filelist_training_file_name, filelist_validation_file_name, get_symbols_path, get_filelist_dir, get_checkpoint_dir, get_log_dir, filelist_weights_file_name
from src.common.train_log import log

from src.tacotron.train import load_model, prepare_directories_and_logger, validate_core, prepare_dataloaders, load_checkpoint
from src.common.utils import get_last_checkpoint

def init_eval_checkpoints_parser(parser):
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--training_dir', type=str, required=True)
  parser.add_argument('--speakers', type=str, required=True)
  parser.add_argument('--hparams', type=str)
  parser.add_argument('--select', type=int)
  parser.add_argument('--min_it', type=int)
  parser.add_argument('--max_it', type=int)
  return __eval_chkpoints

def __eval_chkpoints(hparams, training_dir_path, select: int, min_it: int, max_it: int):
  n_gpus = 1
  filelist_dir_path = get_filelist_dir(training_dir_path)
  _, valset, collate_fn = prepare_dataloaders(hparams, filelist_dir_path)
  
  checkpoint_dir = get_checkpoint_dir(training_dir_path)
  _, _, checkpoints = next(os.walk(checkpoint_dir))
  checkpoints = list(sorted(list(map(int, checkpoints))))
  
  print("Available checkpoints")
  print(checkpoints)

  if not select:
    select = 0
  if not min_it:
    min_it = 0
  if not max_it:
    max_it = max(checkpoints)
  process_checkpoints = [checkpoint for checkpoint in checkpoints if checkpoint % select == 0 and checkpoint >= min_it and checkpoint <= max_it]
  if len(process_checkpoints) == 0:
    print("None selected. Exiting.")
    return
  print("Selected checkpoints")
  print(process_checkpoints)
  result = []
  print("Validating...")
  for checkpoint in tqdm(process_checkpoints):
    criterion = Tacotron2Loss()
    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)
    model = load_model(hparams)
    learning_rate = hparams.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=hparams.weight_decay)
    full_checkpoint_path = os.path.join(checkpoint_dir, str(checkpoint))
    model, _, _, _ = load_checkpoint(full_checkpoint_path, model, optimizer, training_dir_path)
    val_loss, _, _, _ = validate_core(model, criterion, valset, hparams.batch_size, n_gpus, collate_fn, hparams.distributed_run)
    result.append((checkpoint, val_loss))
    print("Validation loss {}: {:9f}".format(checkpoint, val_loss))

  result.sort()
  print("Result...")
  print("Sorted after checkpoints:")
  for cp, loss in result:
    print("Validation loss {}: {:9f}".format(cp, loss))

  result = [(b, a) for a, b in result]
  result.sort()

  print()
  print("Sorted after scores:")
  for loss, cp in result:
    print("Validation loss {}: {:9f}".format(cp, loss))

def main(base_dir: str, training_dir: str, speakers: str, hparams: str, select: int, min_it: int, max_it: int):
  hparams = create_hparams(hparams)
  training_dir_path = os.path.join(base_dir, training_dir)

  conv = load_from_file(get_symbols_path(training_dir_path))
  
  hparams.n_symbols = conv.get_symbol_ids_count()
  n_speakers = len(parse_ds_speakers(speakers))
  hparams.n_speakers = n_speakers

  __eval_chkpoints(hparams, training_dir_path, select=select, min_it=min_it, max_it=max_it)

if __name__ == "__main__":
  main(
    base_dir = '/datasets/models/taco2pt_v2',
    speakers = 'thchs_v5,B2;thchs_v5,A2',
    training_dir = 'debug_ljs_ms',
    hparams = 'batch_size=20',
    select = 500,
    min_it = 0,
  )
