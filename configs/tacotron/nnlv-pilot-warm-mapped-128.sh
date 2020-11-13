# Init
## Capslock
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
source /datasets/code/tacotron2/configs/envs/caps.sh
export train_name="nnlv_pilot_warm_mapped_128"
export prep_name="nnlv_pilot_ipa"
export batch_size=21
export epochs_per_checkpoint=25

## Phil
source /home/stefan/tacotron2/configs/envs/phil.sh
export train_name="nnlv_pilot_warm_mapped_128"
export prep_name="nnlv_pilot_ipa"
export batch_size=17
export epochs_per_checkpoint=50

# Create Weights Map
python -m src.cli.runner prepare-weights-map \
  --weights_prep_name="ljs_ipa" \
  --prep_name=$prep_name

# Training
python -m src.cli.runner tacotron-train \
  --train_name=$train_name \
  --prep_name=$prep_name \
  --test_size=0.001 \
  --validation_size=0.01 \
  --warm_start_train_name="ljs_ipa_scratch_128" \
  --weights_train_name="ljs_ipa_scratch_128" \
  --map_from_speaker="ljs,1" \
  --use_weights_map \
  --custom_hparams="batch_size=$batch_size,iters_per_checkpoint=0,epochs_per_checkpoint=$epochs_per_checkpoint"

python -m src.cli.runner tacotron-continue-train --train_name=$train_name
# Inference

python -m src.cli.runner tacotron-validate --train_name=$train_name

export text_name="ger-nord"
export text_name="eng-democritus"
export text_name="chn-north"
export speaker="nnlv_pilot,phd1"

python -m src.cli.runner tacotron-infer \
  --train_name=$train_name \
  --speaker=$speaker \
  --text_name=$text_name \
  --analysis
