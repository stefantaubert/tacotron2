########################################################################################
# Waveglow from scratch
########################################################################################

# Init
## Capslock
source /datasets/code/tacotron2/configs/envs/caps.sh
export batch_size=4
export iters_per_checkpoint=2500

## GCP
source /home/stefan_taubert/tacotron2/configs/envs/gcp.sh

## Phil
source /home/stefan/tacotron2/configs/envs/phil.sh
export batch_size=3
export iters_per_checkpoint=5000 # 17min per 1000

# Training
python -m src.cli.runner waveglow-train \
  --train_name="scratch" \
  --prep_name="thchs_ljs_ipa" \
  --test_size=0.001 \
  --validation_size=0.01 \
  --custom_hparams="batch_size=$batch_size,iters_per_checkpoint=$iters_per_checkpoint,epochs=100000"

## Continue training
python -m src.cli.runner waveglow-continue-train \
  --train_name="scratch"

# Validate
python -m src.cli.runner waveglow-validate \
  --train_name="scratch"

# Inference
python -m src.cli.runner waveglow-infer \
  --train_name="scratch" \
  --wav_path="$ljs_data/wavs/LJ001-0098.wav"
