#################
# Preprocessing #
#################

# Init

## Capslock
source /datasets/code/tacotron2/configs/envs/caps.sh

## Phil
source /home/stefan/tacotron2/configs/envs/phil.sh

# Preprocessing NNLV Pilot

export ds_name="nnlv_pilot"
python -m src.cli.runner preprocess-custom \
  --path="$nnlv_pilot_data" \
  --ds_name=$ds_name

python -m src.cli.runner preprocess-text \
  --ds_name=$ds_name \
  --text_name="en"

python -m src.cli.runner text-normalize \
  --ds_name=$ds_name \
  --orig_text_name="en" \
  --dest_text_name="en_norm"

python -m src.cli.runner text-ipa \
  --ds_name=$ds_name \
  --orig_text_name="en_norm" \
  --dest_text_name="ipa_norm" \
  --ignore_tones \
  --ignore_arcs

python -m src.cli.runner preprocess-wavs \
  --ds_name=$ds_name \
  --wav_name="96000Hz_stereo"

python -m src.cli.runner wavs-normalize \
  --ds_name=$ds_name \
  --orig_wav_name="96000Hz_stereo" \
  --dest_wav_name="96000Hz_norm_stereo"

python -m src.cli.runner wavs-stereo-to-mono \
  --ds_name=$ds_name \
  --orig_wav_name="96000Hz_norm_stereo" \
  --dest_wav_name="96000Hz_norm_mono"

python -m src.cli.runner wavs-resample \
  --ds_name=$ds_name \
  --orig_wav_name="96000Hz_norm_mono" \
  --dest_wav_name="22050Hz_norm_mono" \
  --rate=22050

python -m src.cli.runner preprocess-mels \
  --ds_name=$ds_name \
  --wav_name="22050Hz_norm_mono"

export prep_name="nnlv_pilot_ipa"
python -m src.cli.runner prepare-ds \
  --prep_name=$prep_name \
  --ds_speakers="$ds_name,all" \
  --ds_text_audio="$ds_name,ipa_norm,22050Hz_norm_mono"
