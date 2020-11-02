# Init
## Capslock
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
source /datasets/code/tacotron2/configs/envs/caps.sh
export train_name="thchs_ipa_warm_mapped_w_tones_speaker_mapped_D4_D6_64"
export prep_name="thchs_ipa_D4_D6"
export audio_name="22050Hz_normalized_nosil"
export batch_size=21
export epochs_per_checkpoint=10

## Phil
source /home/stefan/tacotron2/configs/envs/phil.sh
export train_name="thchs_ipa_warm_mapped_w_tones_speaker_mapped_D4_D6_64"
export prep_name="thchs_ipa_D4_D6"
export audio_name="22050Hz_norm_wo_sil"
export batch_size=17
export epochs_per_checkpoint=10


python -m src.cli.runner prepare-ds \
  --prep_name=$prep_name \
  --ds_speakers="thchs,D4;thchs,D6" \
  --ds_text_audio="thchs,ipa,$audio_name"

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


export text_name="chn-north"
export speaker="thchs,D4"
export speaker="thchs,D6"

export accent="D4"
export accent="D6"
export accent="none"

python -m src.cli.runner prepare-text-set-accent \
  --prep_name=$prep_name \
  --text_name=$text_name \
  --accent=$accent

python -m src.cli.runner tacotron-infer \
  --train_name=$train_name \
  --speaker=$speaker \
  --text_name=$text_name \
  --analysis

