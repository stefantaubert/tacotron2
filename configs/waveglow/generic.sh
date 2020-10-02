# Download pretrained waveglow model
python -m src.cli.runner waveglow-dl

## Open Tensorboard
# if you get an error message, try: `pip uninstall tensorboard-plugin-wit`
# example error message: `ValueError: Not a TBLoader or TBPlugin subclass: <class 'tensorboard_plugin_wit.wit_plugin_loader.WhatIfToolPluginLoader'>`
# see: https://github.com/tensorflow/tensorboard/issues/3549
tensorboard --logdir=$base_dir/waveglow/$train_name/logs

# Validate
python -m src.cli.runner waveglow-validate --train_name="pretrained_v3"


python -m src.cli.runner waveglow-infer --train_name="pretrained_v3" \
  --wav_path="/home/stefan/taco2pt_v5/pre/ds/arctic/wav/22050Hz/13000-13499/13488.wav"

python -m src.cli.runner waveglow-infer --train_name="pretrained_v3" \
  --wav_path="/datasets/models/taco2pt_v5/pre/ds/arctic/wav/22050Hz/13000-13499/13488.wav"

python -m src.cli.runner waveglow-infer --train_name="pretrained_v3" \
  --wav_path="/home/stefan/datasets/arctic/SVBI/wav/arctic_b0530.wav" \
  --custom_hparams="sampling_rate=44100"

python -m src.cli.runner waveglow-infer --train_name="pretrained_v3" \
  --wav_path="/datasets/l2arctic/SVBI/wav/arctic_b0530.wav" \
  --custom_hparams="sampling_rate=44100"
