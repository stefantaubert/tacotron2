#################
# Preprocessing #
#################

# Init

## Capslock
source /datasets/code/tacotron2/configs/envs/caps.sh

## Phil
source /home/stefan/tacotron2/configs/envs/phil.sh

# Preprocessing Arctic

python -m src.cli.runner preprocess-arctic \
  --path="$arctic_data" \
  --auto_dl \
  --ds_name="arctic"

python -m src.cli.runner preprocess-text \
  --ds_name="arctic" \
  --text_name="en"

python -m src.cli.runner text-normalize \
  --ds_name="arctic" \
  --orig_text_name="en" \
  --dest_text_name="en_norm"

python -m src.cli.runner text-ipa \
  --ds_name="arctic" \
  --orig_text_name="en_norm" \
  --dest_text_name="ipa_norm" \
  --ignore_tones \
  --ignore_arcs

python -m src.cli.runner preprocess-wavs \
  --ds_name="arctic" \
  --wav_name="44100Hz"

python -m src.cli.runner preprocess-mels \
  --ds_name="arctic" \
  --wav_name="44100Hz" \
  --custom_hparams="sampling_rate=44100"

python -m src.cli.runner prepare-ds \
  --prep_name="arctic_ipa" \
  --ds_speakers="arctic,all" \
  --ds_text_audio="arctic,ipa_norm,44100Hz"

python -m src.cli.runner wavs-resample \
  --ds_name="arctic" \
  --orig_wav_name="44100Hz" \
  --dest_wav_name="22050Hz" \
  --rate=22050

python -m src.cli.runner preprocess-mels \
  --ds_name="arctic" \
  --wav_name="22050Hz"

export prep_name="arctic_ipa_22050"
python -m src.cli.runner prepare-ds \
  --prep_name=$prep_name \
  --ds_speakers="arctic,all" \
  --ds_text_audio="arctic,ipa_norm,22050Hz"
