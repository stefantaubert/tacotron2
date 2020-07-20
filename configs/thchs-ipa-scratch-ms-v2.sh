########################################################################################
# THCHS-30 based IPA Synthesis
########################################################################################

# Init

## Capslock Dev
source /datasets/code/tacotron2-dev/configs/envs/dev-caps.sh
export custom_training_name="thchs_chn_ipa_scratch_ms_v2"
export ds_name="thchs_v5"
export speakers="$ds_name,all"
export batch_size=17

## Capslock GCP
source /datasets/code/tacotron2-dev/configs/envs/prod-caps.sh
export custom_training_name="thchs_chn_ipa_scratch_ms_v2"
export ds_name="thchs_v5"
export speakers="$ds_name,all"
export batch_size=17

## GCP
# For usage with a t4 on Google Cloud Plattform
source /home/stefan_taubert/tacotron2/configs/envs/prod-gcp.sh
export custom_training_name="thchs_chn_ipa_scratch_ms_v2"
export ds_name="thchs_v5"
export speakers="$ds_name,all"
export batch_size=0

## Phil
source /home/stefan/tacotron2/configs/envs/prod-phil.sh
export custom_training_name="thchs_chn_ipa_scratch_ms_v2"
export ds_name="thchs_ipa"
export speakers="$ds_name,all"
export batch_size=17


# Preprocessing
python script_thchs_pre.py \
  --base_dir=$base_dir \
  --data_dir="$thchs_original_data" \
  --data_conversion_dir="$thchs_data" \
  --ignore_arcs \
  --ignore_tones \
  --auto_dl \
  --kaldi_version \
  --auto_convert \
  --ds_name=$ds_name \
  --no_debugging

# Training from scratch
export hparams="batch_size=$batch_size,iters_per_checkpoint=500,epochs=500"
python -m paths \
  --base_dir=$base_dir \
  --custom_training_name=$custom_training_name \
  --no_debugging
python -m script_train \
  --base_dir=$base_dir \
  --training_dir=$custom_training_name \
  --speakers=$speakers \
  --hparams=$hparams \
  --train_size=0.94 \
  --validation_size=1.0 \
  --no_debugging

## Continue training
export hparams="batch_size=$batch_size,iters_per_checkpoint=500,epochs=500"
python -m script_train \
  --base_dir=$base_dir \
  --training_dir=$custom_training_name \
  --hparams=$hparams \
  --continue_training \
  --no_debugging

# Inference
python -m script_dl_waveglow_pretrained \
  --pretrained_dir=$pretrained_dir \
  --no_debugging
export text_map="maps/inference/chn_v1.json"
export speaker="$ds_name,B8"

# Validate
python script_dl_waveglow_pretrained.py \
  --pretrained_dir=$pretrained_dir \
  --no_debugging

export utterance="A23_75"
python script_validate.py --no_debugging --base_dir=$base_dir --training_dir=$custom_training_name --waveglow=$waveglow --utterance=$utterance --custom_checkpoint='80000'

