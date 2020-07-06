import os
import time
import argparse
import math
from numpy import finfo
from tqdm import tqdm
import numpy as np

import torch
from distributed import apply_gradient_allreduce
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import argparse
from torch.utils.data import DataLoader

from model import Tacotron2
from data_utils import SymbolsMelLoader, SymbolsMelCollate
from loss_function import Tacotron2Loss
from logger import Tacotron2Logger
from hparams import create_hparams
from utils import parse_ds_speakers, get_total_duration_min_df

from text.symbol_converter import load_from_file
from paths import filelist_training_file_name, filelist_validation_file_name, get_symbols_path, get_filelist_dir, get_checkpoint_dir, get_log_dir, filelist_weights_file_name
from train_log import log

from train import load_model, prepare_directories_and_logger, get_last_checkpoint, validate_core, prepare_dataloaders, load_checkpoint

def eval_chkpoints(hparams, training_dir_path):
  n_gpus = 1
  criterion = Tacotron2Loss()
  torch.manual_seed(hparams.seed)
  torch.cuda.manual_seed(hparams.seed)

  filelist_dir_path = get_filelist_dir(training_dir_path)

  model = load_model(hparams)
  learning_rate = hparams.learning_rate
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=hparams.weight_decay)
  _, valset, collate_fn = prepare_dataloaders(hparams, filelist_dir_path)
  
  checkpoint_dir = get_checkpoint_dir(training_dir_path)
  _, _, checkpoints = next(os.walk(checkpoint_dir))
  checkpoints = list(sorted(list(map(int, checkpoints))))
  result = []
  for checkpoint in tqdm(checkpoints):
    full_checkpoint_path = os.path.join(checkpoint_dir, str(checkpoint))
    model, _, _, _ = load_checkpoint(full_checkpoint_path, model, optimizer, training_dir_path)
    val_loss, _, _, _ = validate_core(model, criterion, valset, hparams.batch_size, n_gpus, collate_fn, hparams.distributed_run)
    result.append((checkpoint, val_loss))

  result.sort()
  print("Sorted after checkpoints:")
  for cp, loss in result:
    print("Validation loss {}: {:9f}".format(cp, loss))

  result = [(b, a) for a, b in result]
  result.sort()

  print()
  print("Sorted after scores:")
  for loss, cp in result:
    print("Validation loss {}: {:9f}".format(cp, loss))

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--no_debugging', action='store_true')
  parser.add_argument('--base_dir', type=str, help='base directory')
  parser.add_argument('--training_dir', type=str)
  parser.add_argument('--speakers', type=str)
  parser.add_argument('--hparams', type=str)

  args = parser.parse_args()

  if not args.no_debugging:
    args.base_dir = '/datasets/models/taco2pt_v2'
    args.speakers = 'thchs_v5,B2;thchs_v5,A2'
    args.training_dir = 'debug_ljs_ms'
    args.hparams = 'batch_size=20'


  hparams = create_hparams(args.hparams)
  training_dir_path = os.path.join(args.base_dir, args.training_dir)


  conv = load_from_file(get_symbols_path(training_dir_path))
  
  hparams.n_symbols = conv.get_symbol_ids_count()
  n_speakers = len(parse_ds_speakers(args.speakers))
  hparams.n_speakers = n_speakers


  eval_chkpoints(hparams, training_dir_path)