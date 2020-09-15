import torch

from src.core.tacotron.hparams import create_hparams
from src.core.tacotron.training import Checkpoint, save_checkpoint


def convert(old_model_path: str, new_model_path: str, custom_hparams: str, speakers, accents, symbols):

  checkpoint_dict = torch.load(old_model_path, map_location='cpu')
  hp = create_hparams(
    n_speakers=len(speakers),
    n_accents=len(accents),
    n_symbols=len(symbols),
    hparams_string=custom_hparams
  )

  chp = Checkpoint(
    state_dict=checkpoint_dict["state_dict"],
    optimizer=checkpoint_dict["optimizer"],
    learning_rate=checkpoint_dict["learning_rate"],
    iteration=checkpoint_dict["iteration"],
    hparams=hp,
    speakers=speakers,
    symbols=symbols,
    accents=accents
  )

  save_checkpoint(new_model_path, chp)
