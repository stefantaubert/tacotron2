########################################################################################
# LJSpeech based IPA Synthesis
########################################################################################

# Init

## Capslock Dev
source /datasets/code/tacotron2-dev/configs/envs/dev-caps.sh
export custom_training_name="ljs_ipa_ms_from_scratch"
export ds_name="ljs_ipa_v2"
export speakers="$ds_name,all"
export batch_size=26

## Capslock GCP
source /datasets/code/tacotron2-dev/configs/envs/prod-caps.sh
export custom_training_name="ljs_ipa_ms_from_scratch"
export ds_name="ljs_ipa_v2"
export speakers="$ds_name,all"
export batch_size=26

## GCP
# For usage with a t4 on Google Cloud Plattform
source /home/stefan_taubert/tacotron2/configs/envs/prod-gcp.sh
export custom_training_name="ljs_ipa_ms_from_scratch"
export ds_name="ljs_ipa"
export speakers="$ds_name,all"
export batch_size=52

## Phil
source /home/stefan/tacotron2/configs/envs/prod-phil.sh
export custom_training_name="ljs_ipa_ms_from_scratch"
export ds_name="ljs_ipa"
export speakers="$ds_name,all"
export batch_size=26


# Preprocessing
python ./src/script_ljs_pre.py \
  --base_dir=$base_dir \
  --data_dir="$ljs_data" \
  --ipa \
  --ignore_arcs \
  --ds_name=$ds_name \
  --auto_dl \
  --no_debugging


# Training from scratch
export hparams="batch_size=$batch_size,iters_per_checkpoint=500,epochs=1000"
python ./src/script_paths.py \
  --base_dir=$base_dir \
  --custom_training_name=$custom_training_name \
  --no_debugging
python ./src/tacotron/script_train.py \
  --base_dir=$base_dir \
  --training_dir=$custom_training_name \
  --speakers=$speakers \
  --hparams=$hparams \
  --train_size=0.9 \
  --validation_size=1.0 \
  --no_debugging


## Continue training
export hparams="batch_size=$batch_size,iters_per_checkpoint=500,epochs=1000"
python ./src/tacotron/script_train.py \
  --base_dir=$base_dir \
  --training_dir=$custom_training_name \
  --hparams=$hparams \
  --continue_training \
  --no_debugging


# Inference
python ./src/tacotron/script_dl_waveglow_pretrained.py \
  --destination=$waveglow \
  --auto_convert \
  --no_debugging

export text_map="maps/inference/en_v1.json"
export speaker="$ds_name,1"

# Validate
# export utterance="LJ002-0205"
# export utterance="LJ006-0229"
# export utterance="LJ027-0076"
# last valid checkpoint: 113204
