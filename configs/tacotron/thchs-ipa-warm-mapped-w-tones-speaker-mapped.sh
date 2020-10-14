# Init
## Capslock
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
source /datasets/code/tacotron2/configs/envs/caps.sh
export train_name="thchs_ipa_warm_mapped_w_tones_speaker_mapped"
export prep_name="thchs"
export batch_size=17
export epochs_per_checkpoint=1nɔɹθ wɪnd ɡejv ʌp ðə ətɛmpt. ðɛn ðə sʌn ʃajnd awt wɔɹmli, ænd ɪmidiətli ðə tɹævəlɹ̩ tʊk ɔf hɪz klowk. ænd sow ðə nɔɹθ wɪnd wɑz əblajdʒd tə kənfɛs ðæt ðə sʌn wɑz ðə stɹɔŋɹ̩ ʌv ðə t

## Phil
source /home/stefan/tacotron2/configs/envs/phil.sh
export train_name="thchs_ipa_warm_mapped_w_tones_speaker_mapped"
export prep_name="thchs_ipa"
export batch_size=17
export epochs_per_checkpoint=1

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
  --warm_start_train_name="ljs_ipa_scratch_128" \
  --weights_train_name="ljs_ipa_scratch_128" \
  --map_from_speaker="ljs,1" \
  --use_weights_map \
  --custom_hparams="batch_size=$batch_size,iters_per_checkpoint=0,epochs_per_checkpoint=$epochs_per_checkpoint"

python -m src.cli.runner tacotron-continue-train --train_name=$train_name
# Inference

python -m src.cli.runner tacotron-validate --train_name=$train_name --custom_tacotron_hparams="max_decoder_steps=2000"

# Update Inference Map
python -m src.cli.runner prepare-inference-map \
  --prep_name=$prep_name \
  --template_map="maps/inference/chn_ipa.json"
  #--template_map="maps/weights/thchs_ipa_ljs_ipa.json"


export ds_speaker="thchs,D31"
export ds_speaker="thchs,D4"

python -m src.cli.runner tacotron-infer \
  --train_name=$train_name \
  --ds_speaker=$ds_speaker \
  --text_name=$text_name \
  --custom_tacotron_hparams="max_decoder_steps=2000" \
  --analysis

