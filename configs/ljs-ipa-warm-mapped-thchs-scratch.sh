########################################################################################
# LJSpeech based IPA Synthesis
########################################################################################
# For usage with a t4 on Google Cloud Plattform

# Init

## Capslock Dev
source /datasets/code/tacotron2/configs/envs/dev-caps.sh
export custom_training_name="ljs_ipa_warm_mapped_scratch"
export ds_name="ljs_ipa_v2"
export speakers="$ds_name,all"
export batch_size=26

## Capslock GCP
source /datasets/code/tacotron2/configs/envs/prod-caps.sh
export custom_training_name="ljs_ipa_warm_mapped_scratch"
export ds_name="ljs_ipa_v2"
export speakers="$ds_name,all"
export batch_size=26

## GCP
# For usage with a t4 on Google Cloud Plattform
source /home/stefan_taubert/tacotron2/configs/envs/prod-gcp.sh
export custom_training_name="ljs_ipa_warm_mapped_scratch"
export ds_name="ljs_ipa"
export speakers="$ds_name,all"
export batch_size=52

## Phil
source /home/stefan/tacotron2/configs/envs/prod-phil.sh
export custom_training_name="ljs_ipa_warm_mapped_scratch"
export ds_name="ljs_ipa"
export speakers="$ds_name,all"
export batch_size=26


# Preprocessing
python -m src.runner ljs-pre \
  --base_dir=$base_dir \
  --data_dir="$ljs_data" \
  --ipa \
  --ignore_arcs \
  --ds_name=$ds_name \
  --auto_dl

# Training
export hparams="batch_size=$batch_size,iters_per_checkpoint=500,epochs=500,ignore_layers=[embedding.weight,speakers_embedding.weight]"
python -m src.runner paths \
  --base_dir=$base_dir \
  --custom_training_name=$custom_training_name
python -m src.runner tacotron-train
  --base_dir=$base_dir \
  --training_dir=$custom_training_name \
  --speakers=$speakers \
  --hparams=$hparams \
  --validation_size=0.1 \
  --test_size=0 \
  --warm_start \
  --pretrained_path="$base_dir/thchs_ipa_scratch/checkpoints/29000" \
  --pretrained_model="$base_dir/thchs_ipa_scratch/checkpoints/29000" \
  --pretrained_model_symbols="$base_dir/thchs_ipa_scratch/filelist/symbols.json" \
  --weight_map_mode='use_map' \
  --map="maps/weights/en_chn_v1.json"

## Continue training
export hparams="batch_size=$batch_size,iters_per_checkpoint=500,epochs=500"
python -m src.runner tacotron-train
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
