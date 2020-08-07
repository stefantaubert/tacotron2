########################################################################################
########################################################################################

# Init

## Capslock Dev
source /datasets/code/tacotron2/configs/envs/dev-caps.sh
export custom_training_name="ljs_waveglow"
export ds_name="ljs_22050kHz"
export batch_size=4
export iters_per_checkpoint=5
export epochs=1
export hparams="batch_size=$batch_size,iters_per_checkpoint=$iters_per_checkpoint,epochs=$epochs,with_tensorboard=True"

## Capslock GCP
source /datasets/code/tacotron2/configs/envs/prod-caps.sh

## GCP
# For usage with a t4 on Google Cloud Plattform
source /home/stefan_taubert/tacotron2/configs/envs/prod-gcp.sh

## Phil
source /home/stefan/tacotron2/configs/envs/prod-phil.sh
export custom_training_name="ljs_waveglow"
export ds_name="ljs_22050kHz"
export batch_size=3
export iters_per_checkpoint=5000 # 17min per 1000
export epochs=100000
export hparams="batch_size=$batch_size,iters_per_checkpoint=$iters_per_checkpoint,epochs=$epochs"


# Preprocessing
python -m src.runner ljs-wavs \
  --path="$ljs_data" \
  --auto_dl \
  --base_dir=$base_dir \
  --name=$ds_name


# Training
python -m src.runner waveglow-train \
  --base_dir=$base_dir \
  --training_dir=$custom_training_name \
  --wav_ds_name=$ds_name \
  --test_size=0.001 \
  --validation_size=0.01 \
  --hparams=$hparams


## Continue training
python -m src.runner waveglow-train \
  --base_dir=$base_dir \
  --training_dir=$custom_training_name \
  --hparams=$hparams \
  --continue_training


## Validate
python -m src.runner waveglow-validate \
  --base_dir=$base_dir \
  --training_dir=$custom_training_name \
  --utterance="random-val" \
  --hparams=$hparams \
  --denoiser_strength=0.0 \
  --sigma=1.0 \
  --custom_checkpoint=''


## Inference
python -m src.runner waveglow-infer \
  --base_dir=$base_dir \
  --training_dir=$custom_training_name \
  --wav="$ljs_data/wavs/LJ001-0098.wav" \
  --hparams=$hparams \
  --denoiser_strength=0.0 \
  --sigma=1.0 \
  --custom_checkpoint=0

