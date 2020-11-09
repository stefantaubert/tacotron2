#################
# Preprocessing #
#################

# Init

## Capslock
source /datasets/code/tacotron2/configs/envs/caps.sh

## Phil
source /home/stefan/tacotron2/configs/envs/phil.sh


# Preprocessing LibriTTS

python -m src.cli.runner preprocess-libritts \
  --path="$libritts_data" \
  --auto_dl \
  --ds_name="libritts"

python -m src.cli.runner preprocess-text \
  --ds_name="libritts" \
  --text_name="en"

python -m src.cli.runner text-normalize \
  --ds_name="libritts" \
  --orig_text_name="en" \
  --dest_text_name="en_norm"

python -m src.cli.runner preprocess-wavs \
  --ds_name="libritts" \
  --wav_name="24000Hz"

# 4h47min
python -m src.cli.runner wavs-resample \
  --ds_name="libritts" \
  --orig_wav_name="24000Hz" \
  --dest_wav_name="22050Hz" \
  --rate=22050

# 31min
python -m src.cli.runner preprocess-mels \
  --ds_name="libritts" \
  --wav_name="22050Hz"

python -m src.cli.runner text-ipa \
  --ds_name="libritts" \
  --orig_text_name="en_norm" \
  --dest_text_name="ipa_norm" \
  --ignore_tones \
  --ignore_arcs

export prep_name="libritts_ipa_22050"
python -m src.cli.runner prepare-ds \
  --prep_name=$prep_name \
  --ds_speakers="libritts,all" \
  --ds_text_audio="libritts,ipa_norm,22050Hz"
