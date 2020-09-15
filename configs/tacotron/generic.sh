# Continue training
python -m src.cli.runner tacotron-continue-train --train_name=$train_name

# Create Weights Map
python -m src.cli.runner create-weights-map \
  --orig_prep_name="thchs_ipa" \
  --dest_prep_name="ljs_ipa" \
  --existing_map="maps/weights/en_chn_v1.json"

# Update Inference Map
python -m src.cli.runner create-inference-map \
  --prep_name=$prep_name \
  --corpora="examples/ipa/corpora.txt" \
  --is_ipa \
  --ignore_tones \
  --ignore_arcs \
  --existing_map="maps/inference/$prep_name.json"

# Plot Embeddings
python -m src.cli.runner tacotron-plot-embeddings --train_name=$train_name

# Validate
python -m src.cli.runner tacotron-validate --train_name=$train_name
#  --waveglow="scratch"

# Validate checkpoints
python -m src.cli.runner tacotron-eval-checkpoints \
  --train_name=$train_name \
  --custom_hparams=$hparams \
  --select=2000 \
  --min_it=70000

## Open Tensorboard
# if you get an error message, try: `pip uninstall tensorboard-plugin-wit`
# example error message: `ValueError: Not a TBLoader or TBPlugin subclass: <class 'tensorboard_plugin_wit.wit_plugin_loader.WhatIfToolPluginLoader'>`
# see: https://github.com/tensorflow/tensorboard/issues/3549
tensorboard --logdir=$base_dir/tacotron/$train_name/logs

## Debugging
tensorboard --logdir=$base_dir/tacotron/debug/logs
