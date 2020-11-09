#################
# Preprocessing #
#################

# Init

## Capslock
source /datasets/code/tacotron2/configs/envs/caps.sh

## GCP
source /home/stefan_taubert/tacotron2/configs/envs/gcp.sh

## Phil
source /home/stefan/tacotron2/configs/envs/phil.sh

# Preprocessing THCHS-30

## replace preprocess-thchs with preprocess-thchs-kaldi for kaldi version
python -m src.cli.runner preprocess-thchs \
  --path="$thchs_data" \
  --auto_dl \
  --ds_name="thchs"

python -m src.cli.runner preprocess-text \
  --ds_name="thchs" \
  --text_name="chn"

python -m src.cli.runner text-ipa \
  --ds_name="thchs" \
  --orig_text_name="chn" \
  --dest_text_name="ipa" \
  --ignore_arcs

python -m src.cli.runner preprocess-wavs \
  --ds_name="thchs" \
  --wav_name="16000Hz"

python -m src.cli.runner wavs-normalize \
  --ds_name="thchs" \
  --orig_wav_name="16000Hz" \
  --dest_wav_name="16000Hz_norm"

python -m src.cli.runner wavs-remove-silence \
  --ds_name="thchs" \
  --orig_wav_name="16000Hz_norm" \
  --dest_wav_name="16000Hz_norm_wo_sil" \
  --chunk_size=5 \
  --threshold_start=-20 \
  --threshold_end=-30 \
  --buffer_start_ms=100 \
  --buffer_end_ms=150

python -m src.cli.runner wavs-resample \
  --ds_name="thchs" \
  --orig_wav_name="16000Hz_norm_wo_sil" \
  --dest_wav_name="22050Hz_norm_wo_sil" \
  --rate=22050

python -m src.cli.runner preprocess-mels \
  --ds_name="thchs" \
  --wav_name="22050Hz_norm_wo_sil"

export prep_name="thchs_ipa"
python -m src.cli.runner prepare-ds \
  --prep_name=$prep_name \
  --ds_speakers="thchs,all" \
  --ds_text_audio="thchs,ipa,22050Hz_norm_wo_sil"

## Tool for finding out silence removal parameter config
python -m src.cli.runner wavs-remove-silence-plot \
  --ds_name="thchs" \
  --wav_name="16000Hz_norm" \
  --chunk_size=5 \
  --threshold_start=-20 \
  --threshold_end=-30 \
  --buffer_start_ms=100 \
  --buffer_end_ms=150

# LJS+THCHS IPA merge

export prep_name="thchs_ljs_ipa"
python -m src.cli.runner prepare-ds \
  --prep_name="thchs_ljs_ipa" \
  --ds_speakers="thchs,all;ljs,all" \
  --ds_text_audio="thchs,ipa,22050Hz_norm_wo_sil;ljs,ipa_norm,22050Hz"

