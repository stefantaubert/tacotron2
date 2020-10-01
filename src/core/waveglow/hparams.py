from dataclasses import dataclass


@dataclass
class ExperimentHParams():
  fp16_run: bool = False
  epochs: int = 100000
  iters_per_checkpoint: int = 2000
  epochs_per_checkpoint: int = 1
  seed: int = 1234
  # is not usefull
  cache_wavs: bool = False
  # # dist_config
  # dist_backend="nccl",
  # dist_url="tcp://localhost:54321",


@dataclass
class AudioHParams():
  segment_length: int = 16000
  sampling_rate: int = 22050
  # next 5 occur in mel calculation only
  filter_length: int = 1024
  hop_length: int = 256
  win_length: int = 1024
  mel_fmin: float = 0.0
  mel_fmax: float = 8000.0


@dataclass
class ModelHParams():
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
class HParams(ExperimentHParams, AudioHParams, ModelHParams, OptimizerHParams):
  pass
