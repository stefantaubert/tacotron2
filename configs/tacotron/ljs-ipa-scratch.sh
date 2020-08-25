########################################################################################
# Waveglow from scratch
########################################################################################

# Init
## Capslock
source /datasets/code/tacotron2/configs/envs/caps.sh
export train_name="ljs_ipa_scratch"
export batch_size=26
export iters_per_checkpoint=500
export epochs=1000

## GCP
source /home/stefan_taubert/tacotron2/configs/envs/gcp.sh
export train_name="ljs_ipa_scratch"
export batch_size=52
export iters_per_checkpoint=500
export epochs=1000

## Phil
source /home/stefan/tacotron2/configs/envs/phil.sh
export train_name="ljs_ipa_scratch"
export batch_size=3
export iters_per_checkpoint=500
export epochs=1000

# Training
python -m src.cli.runner tacotron-train \
  --base_dir="$base_dir" \
  --train_name=$train_name \
  --prep_name="ljs_ipa" \
  --test_size=0.001 \
  --validation_size=0.01 \
  --hparams="batch_size=$batch_size,iters_per_checkpoint=$iters_per_checkpoint,epochs=$epochs"

## Continue training
python -m src.cli.runner tacotron-continue-train \
  --base_dir="$base_dir" \
  --train_name=$train_name \
  --hparams="batch_size=$batch_size,iters_per_checkpoint=$iters_per_checkpoint,epochs=$epochs"

# Inference
export text_map="maps/inference/en_ipa.json"
export ds_speaker="ljs,1"

## Update Inference Map
python -m src.cli.runner create-inference-map \
  --base_dir=$base_dir \
  --prep_name="ljs_ipa" \
  --corpora="examples/ipa/corpora.txt" \
  --is_ipa
  --ignore_tones \
  --ignore_arcs \
  --existing_map=$text_map
