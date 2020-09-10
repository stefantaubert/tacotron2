# Download pretrained waveglow model
python -m src.cli.runner waveglow-dl

## Open Tensorboard
# if you get an error message, try: `pip uninstall tensorboard-plugin-wit`
# example error message: `ValueError: Not a TBLoader or TBPlugin subclass: <class 'tensorboard_plugin_wit.wit_plugin_loader.WhatIfToolPluginLoader'>`
# see: https://github.com/tensorflow/tensorboard/issues/3549
tensorboard --logdir=$base_dir/waveglow/$train_name/logs
