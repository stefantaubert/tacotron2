# Init
## Capslock
source /datasets/code/tacotron2/configs/envs/caps.sh
export train_name="arctic_ipa_22050_warm_mapped_64_LXC_HKK_accents"
export prep_name="arctic_ipa_22050_LXC_HKK"
export batch_size=26
export epochs_per_checkpoint=5

## Phil
source /home/stefan/tacotron2/configs/envs/phil.sh
export train_name="arctic_ipa_22050_warm_mapped_64_LXC_HKK_accents"
export prep_name="arctic_ipa_22050_LXC_HKK"
export batch_size=26
export epochs_per_checkpoint=5

python -m src.cli.runner prepare-ds \
  --prep_name=$prep_name \
  --ds_speakers="arctic,LXC;arctic,HKK" \
  --ds_text_audio="arctic,ipa_norm,22050Hz"

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
  --warm_start_train_name="ljs_ipa_scratch_64_accents" \
  --weights_train_name="ljs_ipa_scratch_64_accents" \
  --map_from_speaker="ljs,1" \
  --map_from_accent="north_america" \
  --use_weights_map \
  --custom_hparams="batch_size=$batch_size,iters_per_checkpoint=0,epochs_per_checkpoint=$epochs_per_checkpoint,speakers_embedding_dim=64,accents_embedding_dim=128"

python -m src.cli.runner tacotron-continue-train --train_name=$train_name --custom_hparams="epochs_per_checkpoint=$epochs_per_checkpoint"

python -m src.cli.runner tacotron-validate --train_name=$train_name


# Inference

## add texts...

# Update Inference Map
python -m src.cli.runner prepare-inference-map \
  --prep_name=$prep_name \
  --template_map="maps/inference/eng_ipa.json"
  #--template_map="maps/weights/thchs_ipa_ljs_ipa.json"

export text_name="eng-north"
export text_name="eng-democritus"

export speaker="arctic,LXC"
export speaker="arctic,HKK"

export accent="Chinese-LXC"
export accent="Korean-HKK"


python -m src.cli.runner prepare-text-set-accent \
  --prep_name=$prep_name \
  --text_name=$text_name \
  --accent=$accent

python -m src.cli.runner tacotron-infer \
  --train_name=$train_name \
  --speaker=$speaker \
  --text_name=$text_name \
  --analysis

