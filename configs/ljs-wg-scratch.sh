########################################################################################
# LJSpeech based IPA Synthesis
########################################################################################

# Init

## Capslock Dev
source /datasets/code/tacotron2/configs/envs/dev-caps.sh
export custom_training_name="ljs_waveglow"
export ds_name="ljs_en_v2"
export speakers="$ds_name,all"
export batch_size=4
export iters_per_checkpoint=3
export epochs=1
export hparams="batch_size=$batch_size,iters_per_checkpoint=$iters_per_checkpoint,epochs=$epochs"

## Capslock GCP
source /datasets/code/tacotron2/configs/envs/prod-caps.sh
export custom_training_name="ljs_waveglow"
export ds_name="ljs_en_v2"
export speakers="$ds_name,all"
export batch_size=4
export iters_per_checkpoint=5000
export epochs=100000
export hparams="batch_size=$batch_size,iters_per_checkpoint=$iters_per_checkpoint,epochs=$epochs"

## GCP
# For usage with a t4 on Google Cloud Plattform
source /home/stefan_taubert/tacotron2/configs/envs/prod-gcp.sh
export custom_training_name="ljs_waveglow"
export ds_name="ljs_en_v2"
export speakers="$ds_name,all"
export batch_size=0
export iters_per_checkpoint=5000
export epochs=100000
export hparams="batch_size=$batch_size,iters_per_checkpoint=$iters_per_checkpoint,epochs=$epochs"

## Phil
source /home/stefan/tacotron2/configs/envs/prod-phil.sh
export custom_training_name="ljs_waveglow"
export ds_name="ljs_en_v2"
export speakers="$ds_name,all"
export batch_size=3
export iters_per_checkpoint=5000 # 17min per 1000
export epochs=100000
export hparams="batch_size=$batch_size,iters_per_checkpoint=$iters_per_checkpoint,epochs=$epochs"


# Preprocessing
python ./src/pre/ljs/script_pre.py \
  --base_dir=$base_dir \
  --data_dir="$ljs_data" \
  --ds_name=$ds_name \
  --auto_dl \
  --no_debugging


# Training
python ./src/script_paths.py \
  --base_dir=$base_dir \
  --custom_training_name=$custom_training_name \
  --no_debugging
python ./src/waveglow/script_train.py \
  --base_dir=$base_dir \
  --training_dir=$custom_training_name \
  --speakers=$speakers \
  --hparams=$hparams \
  --validation_size=0.01 \
  --test_size=0 \
  --no_debugging


## Continue training
python ./src/waveglow/script_train.py \
  --base_dir=$base_dir \
  --training_dir=$custom_training_name \
  --hparams=$hparams \
  --continue_training \
  --no_debugging


## Inference
python ./src/waveglow/script_validate.py \
  --base_dir=$base_dir \
  --training_dir=$custom_training_name \
  --utterance=random-val \
  --hparams=$hparams \
  --denoiser_strength=0 \
  --sigma=1 \
  --custom_checkpoint='' \
  --no_debugging

