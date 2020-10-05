from dataclasses import asdict, dataclass
from logging import Logger

from torch.optim.adam import Adam  # pylint: disable=no-name-in-module

from src.core.common.checkpoint import Checkpoint
from src.core.waveglow.hparams import HParams
from src.core.waveglow.model import WaveGlow


@dataclass
class CheckpointWaveglow(Checkpoint):
  # pylint: disable=arguments-differ
  def get_hparams(self, logger: Logger) -> HParams:
    return super().get_hparams(logger, HParams)

  @classmethod
  def from_instances(cls, model: WaveGlow, optimizer: Adam, hparams: HParams, iteration: int):
    result = cls(
      state_dict=model.state_dict(),
      optimizer=optimizer.state_dict(),
      learning_rate=hparams.learning_rate,
      iteration=iteration,
      hparams=asdict(hparams),
    )
    return result
