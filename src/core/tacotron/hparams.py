from dataclasses import dataclass

import tensorflow as tf


@dataclass
class HParams():
  ################################
  # Experiment Parameters    #
  ################################
  epochs = 500
  iters_per_checkpoint = 1000  # 0 if no saving, 1 for each and so on...
  epochs_per_checkpoint = 1  # 0 if no saving, 1 for each and so on...
  seed = 1234
  dynamic_loss_scaling = True
  fp16_run = False
  distributed_run = False
  dist_backend = "nccl"
  dist_url = "tcp://localhost:54321"
  cudnn_enabled = True
  cudnn_benchmark = False
  ignore_layers = []  # [""] -> to define that it is a list

  ################################
  # Data Parameters       #
  ################################
  load_mel_from_disk = False
  cache_mels = False
  use_saved_mels = True

  ################################
  # Audio Parameters       #
  ################################
  n_mel_channels = 80
  sampling_rate = 22050
  # next 5 occur in mel calculation only
  filter_length = 1024
  hop_length = 256
  win_length = 1024
  mel_fmin = 0.0
  mel_fmax = 8000.0

  ################################
  # Model Parameters       #
  ################################
  n_symbols: int
  symbols_embedding_dim = 512
  n_speakers: int
  speakers_embedding_dim = 128,  # 16
  n_accents: int
  accents_embedding_dim = 512
  accents_use_own_symbols = False

  # Encoder parameters
  encoder_kernel_size = 5
  encoder_n_convolutions = 3
  encoder_embedding_dim = 512

  # Decoder parameters
  n_frames_per_step = 1  # currently only 1 is supported
  decoder_rnn_dim = 1024
  prenet_dim = 256
  max_decoder_steps = 1000
  gate_threshold = 0.5
  p_attention_dropout = 0.1
  p_decoder_dropout = 0.1

  # Attention parameters
  attention_rnn_dim = 1024
  attention_dim = 128

  # Location Layer parameters
  attention_location_n_filters = 32
  attention_location_kernel_size = 31

  # Mel-post processing network parameters
  postnet_embedding_dim = 512
  postnet_kernel_size = 5
  postnet_n_convolutions = 5

  ################################
  # Optimization Hyperparameters #
  ################################
  use_saved_learning_rate = False
  learning_rate = 1e-3
  weight_decay = 1e-6
  grad_clip_thresh = 1.0
  batch_size = 64
  mask_padding = True  # set model's padded outputs to padded values


def create_hparams(n_speakers: int, n_symbols: int, n_accents: int, verbose: bool = False):
  """Create model hyperparameters. Parse nondefault from given string."""

  hparams = tf.contrib.training.HParams(
    ################################
    # Experiment Parameters    #
    ################################
    epochs=500,
    iters_per_checkpoint=1000,  # 0 if no saving, 1 for each and so on...
    epochs_per_checkpoint=1,  # 0 if no saving, 1 for each and so on...
    seed=1234,
    dynamic_loss_scaling=True,
    fp16_run=False,
    distributed_run=False,
    dist_backend="nccl",
    dist_url="tcp://localhost:54321",
    cudnn_enabled=True,
    cudnn_benchmark=False,
    ignore_layers=[""],  # [""] -> to define that it is a list

    ################################
    # Data Parameters       #
    ################################
    load_mel_from_disk=False,
    cache_mels=False,
    use_saved_mels=True,

    ################################
    # Audio Parameters       #
    ################################
    n_mel_channels=80,
    sampling_rate=22050,
    # next 5 occur in mel calculation only
    filter_length=1024,
    hop_length=256,
    win_length=1024,
    mel_fmin=0.0,
    mel_fmax=8000.0,

    ################################
    # Model Parameters       #
    ################################
    n_symbols=n_symbols,
    symbols_embedding_dim=512,
    n_speakers=n_speakers,
    speakers_embedding_dim=128,  # 16,
    n_accents=n_accents,
    accents_embedding_dim=512,
    accents_use_own_symbols=False,

    # Encoder parameters
    encoder_kernel_size=5,
    encoder_n_convolutions=3,
    encoder_embedding_dim=512,

    # Decoder parameters
    n_frames_per_step=1,  # currently only 1 is supported
    decoder_rnn_dim=1024,
    prenet_dim=256,
    max_decoder_steps=1000,
    gate_threshold=0.5,
    p_attention_dropout=0.1,
    p_decoder_dropout=0.1,

    # Attention parameters
    attention_rnn_dim=1024,
    attention_dim=128,

    # Location Layer parameters
    attention_location_n_filters=32,
    attention_location_kernel_size=31,

    # Mel-post processing network parameters
    postnet_embedding_dim=512,
    postnet_kernel_size=5,
    postnet_n_convolutions=5,

    ################################
    # Optimization Hyperparameters #
    ################################
    use_saved_learning_rate=False,
    learning_rate=1e-3,
    weight_decay=1e-6,
    grad_clip_thresh=1.0,
    batch_size=64,
    mask_padding=True  # set model's padded outputs to padded values
  )

  # if hparams_string is not None:
  #   if verbose:
  #     tf.logging.info(f"Parsing command line hparams: {hparams_string}")
  #   hparams.parse(hparams_string)

  if verbose:
    tf.logging.info('Final parsed hparams: %s', hparams.values())

  return hparams
