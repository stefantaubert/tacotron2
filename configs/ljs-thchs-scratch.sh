########################################################################################
# LJSpeech based IPA Synthesis
########################################################################################

# Init

## Capslock Dev
source /datasets/code/tacotron2/configs/envs/dev-caps.sh
export custom_training_name="ljs_thchs_scratch"
export ds_name_ljs="ljs_mel_ipa_v1"
export ds_name_thchs="thchs_mel_ipa_norm_nosil"
export speakers="$ds_name_ljs,all;$ds_name_thchs,all"
export batch_size=17

## Capslock GCP
source /datasets/code/tacotron2/configs/envs/prod-caps.sh

## GCP
# For usage with a t4 on Google Cloud Plattform
source /home/stefan_taubert/tacotron2/configs/envs/prod-gcp.sh

## Phil
source /home/stefan/tacotron2/configs/envs/prod-phil.sh
export custom_training_name="ljs_thchs_scratch"
export ds_name_ljs="ljs_mel_ipa_v1"
export ds_name_thchs="thchs_mel_ipa_norm_nosil"
export speakers="$ds_name_ljs,all;$ds_name_thchs,all"
export batch_size=17

# Preprocessing
python -m src.runner ljs-wavs \
  --path="$ljs_data" \
  --auto_dl \
  --base_dir="$base_dir" \
  --name="ljs_22050kHz"

python -m src.runner calc-mels \
  --base_dir=$base_dir \
  --origin_name="ljs_22050kHz" \
  --destination_name="ljs_22050kHz"

python -m src.runner ljs-text \
  --base_dir=$base_dir \
  --mel_name="ljs_22050kHz" \
  --ds_name=$ds_name_ljs \
  --convert_to_ipa

python -m src.runner thchs-wavs \
  --path="$thchs_data" \
  --auto_dl \
  --base_dir="$base_dir" \
  --name="thchs_16000kHz"
  
python -m src.runner normalize \
  --base_dir=$base_dir \
  --origin_name="thchs_16000kHz" \
  --destination_name="thchs_16000kHz_normalized"
  
python -m src.runner upsample \
  --base_dir=$base_dir \
  --origin_name="thchs_16000kHz_normalized" \
  --destination_name="thchs_22050kHz_normalized"
  
python -m src.runner remove-silence \
  --base_dir=$base_dir \
  --origin_name="thchs_22050kHz_normalized" \
  --destination_name="thchs_22050kHz_normalized_nosil" \
  --chunk_size=5 \
  --threshold_start=-20 \
  --threshold_end=-30 \
  --buffer_start_ms=100 \
  --buffer_end_ms=150

python -m src.runner calc-mels \
  --base_dir=$base_dir \
  --origin_name="thchs_22050kHz_normalized_nosil" \
  --destination_name="thchs_22050kHz_normalized_nosil"

python -m src.runner thchs-text \
  --base_dir=$base_dir \
  --mel_name="thchs_22050kHz_normalized_nosil" \
  --ds_name=$ds_name_thchs

# Training from scratch
export hparams="batch_size=$batch_size,iters_per_checkpoint=500,epochs=1000"

python -m src.runner tacotron-train \
  --base_dir=$base_dir \
  --training_dir=$custom_training_name \
  --seed=1234 \
  --speakers=$speakers \
  --validation_size=0.05 \
  --test_size=0.001 \
  --hparams=$hparams

## Continue training
export hparams="batch_size=$batch_size,iters_per_checkpoint=500,epochs=1000"
python -m src.runner tacotron-train \
  --base_dir=$base_dir \
  --training_dir=$custom_training_name \
  --hparams=$hparams \
  --continue_training


# Inference
python -m src.runner waveglow-dl \
  --destination=$waveglow \
  --auto_convert

export text_map="maps/inference/en_v1.json"
export speaker="$ds_name,1"

# Validate
# export utterance="LJ002-0205"
# export utterance="LJ006-0229"
# export utterance="LJ027-0076"
# last valid checkpoint: 113204
