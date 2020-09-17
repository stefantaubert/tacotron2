import logging

import torch

from src.core.common.accents_dict import AccentsDict
from src.core.common.speakers_dict import SpeakersDict
from src.core.common.symbol_id_dict import SymbolIdDict
from src.core.common.train import get_pytorch_filename, hp_raw
from src.core.tacotron.hparams import create_hparams
from src.core.tacotron.training import CheckpointTacotron


def convert_v1_to_v2_model(old_model_path: str, custom_hparams: str, speakers: SpeakersDict, accents: AccentsDict, symbols: SymbolIdDict):
  checkpoint_dict = torch.load(old_model_path, map_location='cpu')
  hp = create_hparams(
    n_speakers=len(speakers),
    n_accents=len(accents),
    n_symbols=len(symbols),
    hparams_string=custom_hparams
  )

  chp = CheckpointTacotron(
    state_dict=checkpoint_dict["state_dict"],
    optimizer=checkpoint_dict["optimizer"],
    learning_rate=checkpoint_dict["learning_rate"],
    iteration=checkpoint_dict["iteration"] + 1,
    hparams=hp_raw(hp),
    speakers=speakers.raw(),
    symbols=symbols.raw(),
    accents=accents.raw()
  )

  new_model_path = old_model_path + "_" + get_pytorch_filename(chp.iteration)

  chp.save(new_model_path, logging.getLogger())
