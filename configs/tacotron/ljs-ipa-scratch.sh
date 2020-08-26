# Init
## Capslock
source /datasets/code/tacotron2/configs/envs/caps.sh
export train_name="ljs_ipa_scratch"
export prep_name="ljs_ipa"
export batch_size=26
export iters_per_checkpoint=500
export epochs=1000

## GCP
source /home/stefan_taubert/tacotron2/configs/envs/gcp.sh
export train_name="ljs_ipa_scratch"
export prep_name="ljs_ipa"
export batch_size=52
export iters_per_checkpoint=500
export epochs=1000

## Phil
source /home/stefan/tacotron2/configs/envs/phil.sh
export train_name="ljs_ipa_scratch"
export prep_name="ljs_ipa"
export batch_size=3
export iters_per_checkpoint=500
export epochs=1000

# Training
export hparams="batch_size=$batch_size,iters_per_checkpoint=$iters_per_checkpoint,epochs=$epochs"
python -m src.cli.runner tacotron-train \
  --base_dir="$base_dir" \
  --train_name=$train_name \
  --prep_name=$prep_name \
  --test_size=0.001 \
  --validation_size=0.01 \
  --hparams=$hparams

# Inference
export ds_speaker="ljs,1"
