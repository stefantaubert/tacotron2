
# Validate
python -m src.runner waveglow-dl \
  --destination=$waveglow \
  --auto_convert

export utterance="random-val"
python -m src.runner tacotron-validate --base_dir=$base_dir --training_dir=$custom_training_name --waveglow=$waveglow --utterance=$utterance --custom_checkpoint=''

# Inference IPA
export text="examples/ipa/north_sven_orig.txt"
python -m src.runner tacotron-infer --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --lang=ipa --ignore_tones --ignore_arcs --speaker=$speaker --waveglow=$waveglow --map=$text_map --analysis --custom_checkpoint=''

export text="examples/ipa/north_ger.txt"
python -m src.runner tacotron-infer --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --lang=ipa --ignore_tones --ignore_arcs --speaker=$speaker --waveglow=$waveglow --map=$text_map --analysis --custom_checkpoint=''

export text="examples/ipa/north_sven_v2.txt"
python -m src.runner tacotron-infer --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --lang=ipa --ignore_tones --ignore_arcs --speaker=$speaker --waveglow=$waveglow --map=$text_map --analysis --custom_checkpoint=''

# Inference CHN

export text="examples/chn/thchs.txt"
python -m src.runner tacotron-infer --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --lang=chn --ignore_tones --ignore_arcs --speaker=$speaker --waveglow=$waveglow --map=$text_map --analysis --custom_checkpoint=''

# Inference GER

export text="examples/ger/example.txt"
python -m src.runner tacotron-infer --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --lang=ger --ignore_tones --ignore_arcs --speaker=$speaker --waveglow=$waveglow --map=$text_map --analysis --custom_checkpoint=''

export text="examples/ger/nord.txt"
python -m src.runner tacotron-infer --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --lang=ger --ignore_tones --ignore_arcs --speaker=$speaker --waveglow=$waveglow --map=$text_map --analysis --custom_checkpoint=''

# Inference EN

export text="examples/en/north.txt"
python -m src.runner tacotron-infer --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --lang=en --ignore_tones --ignore_arcs --speaker=$speaker --waveglow=$waveglow --map=$text_map --analysis --custom_checkpoint=''

export text="examples/en/democritus_v2.txt"
python -m src.runner tacotron-infer --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --lang=en --ignore_tones --ignore_arcs --speaker=$speaker --waveglow=$waveglow --map=$text_map --analysis --custom_checkpoint=''

# Validate checkpoints
export hparams="batch_size=$batch_size"
export select_pattern=10000
python -m src.runner eval-checkpoints --base_dir=$base_dir --training_dir=$custom_training_name --hparams=$hparams --select=2000 --min=70000

# Plot Embeddings
python -m src.runner plot-embeddings \
  --base_dir=$base_dir \
  --training_dir=$custom_training_name \
  --custom_checkpoint=''

# Create Inference Map
python -m src.runner create-map \
  --a="$base_dir/$custom_training_name/filelist/symbols.json" \
  --b="examples/ipa/corpora.txt" \
  --mode="infer" \
  --ignore_tones \
  --ignore_arcs

# Update Inference Map
python -m src.runner create-map \
  --a="$base_dir/$custom_training_name/filelist/symbols.json" \
  --b="examples/ipa/corpora.txt" \
  --existing_map="maps/inference/en_v1.json"
  --mode="infer" \
  --ignore_tones \
  --ignore_arcs

## Open Tensorboard
# if you get an error message, try: `pip uninstall tensorboard-plugin-wit`
# example error message: `ValueError: Not a TBLoader or TBPlugin subclass: <class 'tensorboard_plugin_wit.wit_plugin_loader.WhatIfToolPluginLoader'>`
# see: https://github.com/tensorflow/tensorboard/issues/3549
tensorboard --logdir=$base_dir/$custom_training_name/logs