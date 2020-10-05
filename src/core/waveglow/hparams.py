from dataclasses import dataclass

from src.core.common.taco_stft import TSTFTHParams


@dataclass
class ExperimentHParams():
  #fp16_run: bool = False
  epochs: int = 100000
  iters_per_checkpoint: int = 2000
  epochs_per_checkpoint: int = 1
  seed: int = 1234
  # is not usefull
  cache_wavs: bool = False
  cudnn_enabled: bool = True
  cudnn_benchmark: bool = False


@dataclass
class ModelHParams():
  segment_length: int = 16000
  n_mel_channels: int = 80
  n_flows: int = 12
  n_group: int = 8
  n_early_every: int = 4
  n_early_size: int = 2

  # WN_config
  n_layers: int = 8
  n_channels: int = 256
  kernel_size: int = 3


@dataclass
class OptimizerHParams():
  learning_rate: float = 1e-4
  sigma: float = 1.0
  batch_size: int = 1


@dataclass
class HParams(ExperimentHParams, TSTFTHParams, ModelHParams, OptimizerHParams):
  pass
