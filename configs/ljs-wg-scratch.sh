########################################################################################
# LJSpeech based IPA Synthesis
########################################################################################

# Init

## Capslock Dev
source /datasets/code/tacotron2-dev/configs/envs/dev-caps.sh
export custom_training_name="ljs_waveglow"
export ds_name="ljs_en_v2"
export speakers="$ds_name,all"
export batch_size=4

## Capslock GCP
source /datasets/code/tacotron2-dev/configs/envs/prod-caps.sh
export custom_training_name="ljs_waveglow"
export ds_name="ljs_en_v2"
export speakers="$ds_name,all"
export batch_size=4

## GCP
# For usage with a t4 on Google Cloud Plattform
source /home/stefan_taubert/tacotron2/configs/envs/prod-gcp.sh
export custom_training_name="ljs_waveglow"
export ds_name="ljs_en_v2"
export speakers="$ds_name,all"
export batch_size=0

## Phil
source /home/stefan/tacotron2/configs/envs/prod-phil.sh
export custom_training_name="ljs_waveglow"
export ds_name="ljs_en_v2"
export speakers="$ds_name,all"
export batch_size=0


# Preprocessing
python ./script_ljs_pre.py \
  --base_dir=$base_dir \
  --data_dir="$ljs_data" \
  --ds_name=$ds_name \
  --auto_dl \
  --no_debugging


# Training
export hparams="batch_size=$batch_size,iters_per_checkpoint=50"
python ./script_paths.py \
  --base_dir=$base_dir \
  --custom_training_name=$custom_training_name \
  --no_debugging
python -m script_train_waveglow \
  --base_dir=$base_dir \
  --training_dir=$custom_training_name \
  --speakers=$speakers \
  --hparams=$hparams \
  --train_size=0.9 \
  --validation_size=1.0 \
  --no_debugging


## Continue training
export hparams="batch_size=$batch_size,iters_per_checkpoint=50"
python -m script_train_waveglow \
  --base_dir=$base_dir \
  --training_dir=$custom_training_name \
  --hparams=$hparams \
  --continue_training \
  --no_debugging


# Inference
# python ./tacotron/script_dl_waveglow_pretrained.py \
#   --pretrained_dir=$pretrained_dir \
#   --no_debugging
# export text_map="maps/inference/en_v1.json"
# export speaker="$ds_name,1"
