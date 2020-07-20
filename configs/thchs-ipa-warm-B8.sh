########################################################################################
# THCHS-30 based IPA Synthesis with Chinese Accents
########################################################################################
# Config: toneless, without arcs
# you have to first train ljs-en-ipa

# Init

## Capslock Dev
source /datasets/code/tacotron2-dev/configs/envs/dev-caps.sh
export custom_training_name="thchs_ipa_warm"
export ds_name="thchs_v5"
export speakers="$ds_name,B8"
export batch_size=0

## Capslock GCP
source /datasets/code/tacotron2-dev/configs/envs/prod-caps.sh
export custom_training_name="thchs_ipa_warm"
export ds_name="thchs_v5"
export speakers="$ds_name,B8"
export batch_size=0

## GCP
# For usage with a t4 on Google Cloud Plattform
source /home/stefan_taubert/tacotron2/configs/envs/prod-gcp.sh
export custom_training_name="thchs_ipa_warm"
export ds_name="thchs_v5"
export speakers="$ds_name,B8"
export batch_size=45

## Phil
source /home/stefan/tacotron2/configs/envs/prod-phil.sh
export custom_training_name="thchs_ipa_warm"
export ds_name="thchs_v5"
export speakers="$ds_name,B8"
export batch_size=0


# Preprocessing
python ./script_thchs_pre.py \
  --base_dir=$base_dir \
  --data_dir="$thchs_original_data" \
  --data_conversion_dir="$thchs_data" \
  --ignore_arcs \
  --ignore_tones \
  --auto_dl \
  --auto_convert \
  --ds_name=$ds_name \
  --no_debugging

# Training
export hparams="batch_size=$batch_size,iters_per_checkpoint=500,epochs=2000,ignore_layers=[embedding.weight,speakers_embedding.weight]"
python ./script_paths.py \
  --base_dir=$base_dir \
  --custom_training_name=$custom_training_name \
  --no_debugging
python ./tacotron/script_train.py \
  --base_dir=$base_dir \
  --training_dir=$custom_training_name \
  --speakers=$speakers \
  --hparams=$hparams \
  --train_size=0.9 \
  --validation_size=1.0 \
  --warm_start \
  --pretrained_path="$base_dir/ljs_ipa_ms_from_scratch/checkpoints/79000" \
  --no_debugging

## Continue training
export hparams="batch_size=$batch_size,iters_per_checkpoint=500,epochs=2000"
python ./tacotron/script_train.py \
  --base_dir=$base_dir \
  --training_dir=$custom_training_name \
  --hparams=$hparams \
  --continue_training \
  --no_debugging

# Inference
python ./tacotron/script_dl_waveglow_pretrained.py \
  --pretrained_dir=$pretrained_dir \
  --no_debugging
export text_map="maps/inference/chn_v1.json"
export speaker="$ds_name,B8"
