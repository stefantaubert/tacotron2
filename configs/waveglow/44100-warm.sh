# Init
## Capslock
source /datasets/code/tacotron2/configs/envs/caps.sh
export train_name="44100_warm"
export batch_size=3
export epochs_per_checkpoint=1

## Phil
source /home/stefan/tacotron2/configs/envs/phil.sh
export train_name="44100_warm"
export batch_size=3
export epochs_per_checkpoint=1

# Training
python -m src.cli.runner waveglow-train \
  --train_name=$train_name \
  --prep_name="arctic_ipa" \
  --test_size=0.0001 \
  --validation_size=0.001 \
  --warm_start_train_name="pretrained_v3" \
  --custom_hparams="batch_size=$batch_size,iters_per_checkpoint=0,epochs_per_checkpoint=$epochs_per_checkpoint,epochs=100000,sampling_rate=44100"

## Continue training
python -m src.cli.runner waveglow-continue-train --train_name=$train_name

# Validate
python -m src.cli.runner waveglow-validate --train_name=$train_name

python -m src.cli.runner waveglow-validate --train_name=$train_name --entry_id=10017 --denoiser_strength=0.00 --sigma=0.777
python -m src.cli.runner waveglow-validate --train_name=$train_name --entry_id=10017 --denoiser_strength=0.00 --sigma=1.0

# Inference
python -m src.cli.runner waveglow-infer \
  --train_name=$train_name \
  --wav_path="$ljs_data/wavs/LJ001-0098.wav"
