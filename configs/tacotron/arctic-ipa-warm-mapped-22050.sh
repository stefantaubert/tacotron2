# Init
## Capslock
source /datasets/code/tacotron2/configs/envs/caps.sh
export train_name="arctic_ipa_22050_warm_mapped_128"
export prep_name="arctic_ipa_22050"
export batch_size=26
export epochs_per_checkpoint=1

## Phil
source /home/stefan/tacotron2/configs/envs/phil.sh
export train_name="arctic_ipa_22050_warm_mapped_128"
export prep_name="arctic_ipa_22050"
export batch_size=25
export epochs_per_checkpoint=5

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
  --custom_hparams="batch_size=$batch_size,iters_per_checkpoint=0,epochs_per_checkpoint=$epochs_per_checkpoint,speakers_embedding_dim=128"

python -m src.cli.runner tacotron-continue-train --train_name=$train_name

python -m src.cli.runner tacotron-validate --train_name=$train_name --custom_tacotron_hparams="max_decoder_steps=2000"


# Inference

## add texts...

# Update Inference Map
python -m src.cli.runner prepare-inference-map \
  --prep_name=$prep_name \
  --template_map="maps/inference/eng_ipa.json"
  #--template_map="maps/weights/thchs_ipa_ljs_ipa.json"

export speaker="arctic,ABA"
export speaker="arctic,ZHAA"
export speaker="arctic,BWC"
export speaker="arctic,LXC"

export accent="Chinese-BWC"
export accent="Chinese-LXC"

export text_name="ipa-north_sven_orig"
export text_name="ipa-north_sven_v2"
export text_name="ipa-north_ger"
export text_name="chn-thchs"
export text_name="chn-north"
export text_name="ger-nord"
export text_name="eng-coma"
export text_name="eng-rainbow"
export text_name="eng-stella"
export text_name="eng-democritus_v2"
export text_name="eng-north"

python -m src.cli.runner prepare-text-set-accent \
  --prep_name=$prep_name \
  --text_name=$text_name \
  --accent=$accent

python -m src.cli.runner tacotron-infer \
  --train_name=$train_name \
  --speaker=$speaker \
  --text_name=$text_name \
  --custom_tacotron_hparams="max_decoder_steps=2000" \
  --analysis

