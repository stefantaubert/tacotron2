########################################################################################
# LJSpeech based IPA Synthesis
########################################################################################

# Init

## Capslock Dev
source /datasets/code/tacotron2/configs/envs/dev-caps.sh
export custom_training_name="ljs_ipa_ms_from_scratch"
export ds_name="ljs_ipa_v2"
export speakers="$ds_name,all"
export batch_size=26

## Capslock GCP
source /datasets/code/tacotron2/configs/envs/prod-caps.sh
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
python -m src.runner ljs-pre \
  --base_dir=$base_dir \
  --data_dir="$ljs_data" \
  --ipa \
  --ignore_arcs \
  --ds_name=$ds_name \
  --auto_dl


# Training from scratch
export hparams="batch_size=$batch_size,iters_per_checkpoint=500,epochs=1000"
python -m src.runner paths \
  --base_dir=$base_dir \
  --custom_training_name=$custom_training_name
python -m src.runner tacotron-train
  --base_dir=$base_dir \
  --training_dir=$custom_training_name \
  --speakers=$speakers \
  --hparams=$hparams \
  --validation_size=0.1 \
  --test_size=0


## Continue training
export hparams="batch_size=$batch_size,iters_per_checkpoint=500,epochs=1000"
python -m src.runner tacotron-train
  --base_dir=$base_dir \
  --training_dir=$custom_training_name \
  --hparams=$hparams \
  --continue_training


# Inference
python -m src.runner waveglow-dl \
  --destination=$waveglow \
  --auto_convert

export text_map="maps/inference/en_v1.json"
export speaker="$ds_name,1"

# Validate
# export utterance="LJ002-0205"
# export utterance="LJ006-0229"
# export utterance="LJ027-0076"
# last valid checkpoint: 113204
