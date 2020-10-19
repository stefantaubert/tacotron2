# Init
## Capslock
source /datasets/code/tacotron2/configs/envs/caps.sh
export train_name="ljs_thchs_ipa_warm_mapped_w_tones"
export prep_name="thchs_ljs"
export batch_size=17
export iters_per_checkpoint=500
export epochs_per_checkpoint=5

## GCP
source /home/stefan_taubert/tacotron2/configs/envs/gcp.sh
export train_name="ljs_thchs_ipa_warm_mapped_w_tones"
export prep_name="thchs_ljs"
export batch_size=52
export iters_per_checkpoint=500
export epochs_per_checkpoint=1

## Phil
source /home/stefan/tacotron2/configs/envs/phil.sh
export train_name="ljs_thchs_ipa_warm_mapped_w_tones"
export prep_name="thchs_ljs"
export batch_size=17
export iters_per_checkpoint=1000
export epochs_per_checkpoint=2

# Create Weights Map
python -m src.cli.runner create-weights-map \
  --orig_prep_name="ljs_ipa" \
  --dest_prep_name=$prep_name \
  --existing_map="maps/weights/thchs_ipa_ljs_ipa.json"

# Training
python -m src.cli.runner tacotron-train \
  --train_name=$train_name \
  --prep_name=$prep_name \
  --test_size=0.001 \
  --validation_size=0.01 \
  --custom_hparams="batch_size=$batch_size,iters_per_checkpoint=$iters_per_checkpoint,epochs_per_checkpoint=$epochs_per_checkpoint,epochs=2000,accents_use_own_symbols=True" \
  --warm_start_train_name="ljs_ipa_warm" \
  --weights_train_name="ljs_ipa_warm" \
  --weights_map="maps/weights/thchs_ljs_ipa_ljs_ipa.json"

# Inference
export speaker="thchs,D31"

python -m src.cli.runner tacotron-validate \
  --train_name=$train_name \
  --speaker=$speaker

421000