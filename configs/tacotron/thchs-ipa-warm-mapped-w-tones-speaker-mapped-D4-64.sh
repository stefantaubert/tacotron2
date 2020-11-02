# Init
## Capslock
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
source /datasets/code/tacotron2/configs/envs/caps.sh
export train_name="thchs_ipa_warm_mapped_w_tones_speaker_mapped_D4_64"
export prep_name="thchs_ipa_D4"
export batch_size=21
export epochs_per_checkpoint=50

## Phil
source /home/stefan/tacotron2/configs/envs/phil.sh
export train_name="thchs_ipa_warm_mapped_w_tones_speaker_mapped_D4_64"
export prep_name="thchs_ipa_D4"
export batch_size=17
export epochs_per_checkpoint=50


python -m src.cli.runner prepare-ds \
  --prep_name=$prep_name \
  --ds_speakers="thchs,D4" \
  --ds_text_audio="thchs,ipa,22050Hz_normalized_nosil"

# Create Weights Map
python -m src.cli.runner prepare-weights-map \
  --weights_prep_name="ljs_ipa" \
  --prep_name=$prep_name \
  --template_map="maps/weights/chn_ipa.json"

# Training
python -m src.cli.runner tacotron-train \
  --train_name=$train_name \
  --prep_name=$prep_name \
  --test_size=0.001 \
  --validation_size=0.01 \
  --warm_start_train_name="ljs_ipa_scratch_64" \
  --weights_train_name="ljs_ipa_scratch_64" \
  --map_from_speaker="ljs,1" \
  --use_weights_map \
  --custom_hparams="batch_size=$batch_size,iters_per_checkpoint=0,epochs_per_checkpoint=$epochs_per_checkpoint,speakers_embedding_dim=64"

python -m src.cli.runner tacotron-continue-train --train_name=$train_name --custom_hparams="epochs_per_checkpoint=$epochs_per_checkpoint"
# Inference

python -m src.cli.runner tacotron-validate --train_name=$train_name

# Update Inference Map
python -m src.cli.runner prepare-inference-map \
  --prep_name=$prep_name \
  --template_map="maps/inference/chn_ipa.json"
# NOTE: set for "," -> " " instead of ""

export text_name="ger-nord"
export text_name="eng-democritus"
export text_name="chn-north"
export speaker="thchs,D4"

python -m src.cli.runner tacotron-infer \
  --train_name=$train_name \
  --speaker=$speaker \
  --text_name=$text_name \
  --analysis

