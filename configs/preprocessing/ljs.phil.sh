#################
# Preprocessing #
#################

# Init

## Capslock
source /datasets/code/tacotron2/configs/envs/caps.sh

## Phil
source /home/stefan/tacotron2/configs/envs/phil.sh

# Preprocessing LJ-Speech

python -m src.cli.runner preprocess-ljs \
  --path="$ljs_data" \
  --auto_dl \
  --ds_name="ljs"

python -m src.cli.runner preprocess-text \
  --ds_name="ljs" \
  --text_name="en"

python -m src.cli.runner text-normalize \
  --ds_name="ljs" \
  --orig_text_name="en" \
  --dest_text_name="en_norm"

python -m src.cli.runner text-ipa \
  --ds_name="ljs" \
  --orig_text_name="en_norm" \
  --dest_text_name="ipa_norm" \
  --ignore_tones \
  --ignore_arcs

python -m src.cli.runner preprocess-wavs \
  --ds_name="ljs" \
  --wav_name="22050Hz"

python -m src.cli.runner preprocess-mels \
  --ds_name="ljs" \
  --wav_name="22050Hz"

export prep_name="ljs_ipa"
python -m src.cli.runner prepare-ds \
  --prep_name=$prep_name \
  --ds_speakers="ljs,all" \
  --ds_text_audio="ljs,ipa_norm,22050Hz"
