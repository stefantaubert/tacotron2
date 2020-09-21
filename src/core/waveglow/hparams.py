from typing import Optional

import tensorflow as tf


def create_hparams(verbose: bool = False) -> tf.contrib.training.HParams:
  """Create model hyperparameters. Parse nondefault from given string."""

  hparams = tf.contrib.training.HParams(
    ################################
    # Experiment Parameters    #
    ################################
    fp16_run=False,
    epochs=100000,
    iters_per_checkpoint=2000,
    epochs_per_checkpoint=1,
    seed=1234,
    # is not usefull
    cache_wavs=False,

    # # dist_config
    # dist_backend="nccl",
    # dist_url="tcp://localhost:54321",

    ################################
    # Audio Parameters       #
    ################################
    segment_length=16000,
    sampling_rate=22050,
    filter_length=1024,
    hop_length=256,
    win_length=1024,
    mel_fmin=0.0,
    mel_fmax=8000.0,

    ################################
    # Model Parameters       #
    ################################
    n_mel_channels=80,
    n_flows=12,
    n_group=8,
    n_early_every=4,
    n_early_size=2,

    # WN_config
    n_layers=8,
    n_channels=256,
    kernel_size=3,

    ################################
    # Optimization Hyperparameters #
    ################################
    learning_rate=1e-4,
    sigma=1.0,
    batch_size=12
  )

  # if hparams_string is not None:
  #   if verbose:
  #     tf.logging.info(f"Parsing command line hparams: {hparams_string}")
  #   hparams.parse(hparams_string)

  if verbose:
    tf.logging.info('Final parsed hparams: %s', hparams.values())

  return hparams
