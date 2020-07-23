########################################################################################
# THCHS-30 based IPA Synthesis
########################################################################################
# old and not tested
# Config: toneless, without arcs, no mapping

# Init

## Capslock Dev
source /datasets/code/tacotron2/configs/envs/dev-caps.sh
export custom_training_name="thchs_ipa_scratch"
export ds_name="thchs_v5"
export speakers="$ds_name,B8;$ds_name,B2;$ds_name,A2"
export batch_size=0

## Capslock GCP
source /datasets/code/tacotron2/configs/envs/prod-caps.sh
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

python ./src/pre/thchs/script_dl.py \
  --no_debugging \
  --data_dir=$thchs_original_data \
  --ds_name=$ds_name

python ./src/pre/thchs/script_upsample.py \
  --no_debugging \
  --data_src_dir=$thchs_original_data \
  --data_dest_dir=$thchs_upsampled_data

python ./src/pre/thchs/script_remove_silence.py \
  --no_debugging \
  --data_src_dir=$thchs_upsampled_data \
  --data_dest_dir=$thchs_nosil_data \
  --chunk_size=5 \
  --threshold_start=-25 \
  --threshold_end=-35 \
  --buffer_start_ms=100 \
  --buffer_end_ms=150

python ./src/pre/thchs/script_pre.py \
  --no_debugging \
  --base_dir=$base_dir \
  --data_dir="$thchs_nosil_data" \
  --ignore_arcs \
  --ignore_tones \
  --ds_name=$ds_name

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
  --validation_size=0.1 \
  --test_size=0 \
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
python ./src/waveglow/script_dl_pretrained.py \
  --destination=$waveglow \
  --auto_convert \
  --no_debugging

export text_map="maps/inference/chn_v1.json"
export speaker="$ds_name,A2"
