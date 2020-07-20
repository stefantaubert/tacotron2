########################################################################################
# THCHS-30 based IPA Synthesis
########################################################################################
# old and not tested
# Config: toneless, without arcs, no mapping

# Init

## Capslock Dev
source /datasets/code/tacotron2-dev/configs/envs/dev-caps.sh
export custom_training_name="thchs_ipa_scratch"
export ds_name="thchs_v5"
export speakers="$ds_name,B8;$ds_name,B2;$ds_name,A2"
export batch_size=0

## Capslock GCP
source /datasets/code/tacotron2-dev/configs/envs/prod-caps.sh
export custom_training_name="thchs_ipa_scratch"
export ds_name="thchs_v5"
export speakers="$ds_name,B8;$ds_name,B2;$ds_name,A2"
export batch_size=0

## GCP
# For usage with a t4 on Google Cloud Plattform
source /home/stefan_taubert/tacotron2/configs/envs/prod-gcp.sh
export custom_training_name="thchs_ipa_scratch"
export ds_name="thchs_v5"
export speakers="$ds_name,B8;$ds_name,B2;$ds_name,A2"
export batch_size=35

## Phil
source /home/stefan/tacotron2/configs/envs/prod-phil.sh
export custom_training_name="thchs_ipa_scratch"
export ds_name="thchs_v5"
export speakers="$ds_name,B8;$ds_name,B2;$ds_name,A2"
export batch_size=0


# Preprocessing
python ./src/script_thchs_pre.py \
  --base_dir=$base_dir \
  --data_dir="$thchs_original_data" \
  --data_conversion_dir="$thchs_data" \
  --ignore_arcs \
  --ignore_tones \
  --auto_dl \
  --auto_convert \
  --ds_name=$ds_name \
  --no_debugging

# Training from scratch
export hparams="batch_size=$batch_size,iters_per_checkpoint=500,epochs=2000"
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
export hparams="batch_size=$batch_size,iters_per_checkpoint=500,epochs=2000"
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

export text_map="maps/inference/chn_v1.json"
export speaker="$ds_name,A2"
