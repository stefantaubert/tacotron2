# Init
## Capslock
source /datasets/code/tacotron2/configs/envs/caps.sh
export train_name="thchs_ipa_warm_mapped_w_tones_acc"
export prep_name="thchs_ipa_acc"
export batch_size=17
export iters_per_checkpoint=500
export epochs_per_checkpoint=1
export epochs=2000

## GCP
source /home/stefan_taubert/tacotron2/configs/envs/gcp.sh
export train_name="thchs_ipa_warm_mapped_w_tones_acc"
export prep_name="thchs_ipa_acc"
export batch_size=52
export iters_per_checkpoint=500
export epochs_per_checkpoint=1
export epochs=1000

## Phil
source /home/stefan/tacotron2/configs/envs/phil.sh
export train_name="thchs_ipa_warm_mapped_w_tones_acc"
export prep_name="thchs_ipa_acc"
export batch_size=17
export iters_per_checkpoint=500
export epochs_per_checkpoint=1
export epochs=1000

# Create Weights Map
python -m src.cli.runner create-weights-map \
  --base_dir=$base_dir \
  --orig_prep_name="ljs_ipa" \
  --dest_prep_name=$prep_name \
  --existing_map="maps/weights/thchs_ipa_ljs_ipa.json"

# Training
export hparams="batch_size=$batch_size,iters_per_checkpoint=$iters_per_checkpoint,epochs_per_checkpoint=$epochs_per_checkpoint,epochs=$epochs"
python -m src.cli.runner tacotron-train \
  --base_dir="$base_dir" \
  --train_name=$train_name \
  --prep_name=$prep_name \
  --test_size=0.01 \
  --validation_size=0.1 \
  --hparams=$hparams \
  --warm_start_train_name="ljs_ipa_scratch" \
  --weights_train_name="ljs_ipa_scratch" \
  --weights_map="maps/weights/"$prep_name"_ljs_ipa.json"

# Inference
export ds_speaker="thchs,D31"
