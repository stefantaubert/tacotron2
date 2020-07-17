########################################################################################
# LJSpeech based IPA Synthesis
########################################################################################
# For usage with a t4 on Google Cloud Plattform

# Init

## Capslock Dev
source /datasets/code/tacotron2-dev/configs/envs/dev-caps.sh
export custom_training_name="ljs_ipa_warm_mapped"
export ds_name="ljs_ipa_v2"
export speakers="$ds_name,all"
export batch_size=26

## Capslock GCP
source /datasets/code/tacotron2-dev/configs/envs/prod-caps.sh
export custom_training_name="ljs_ipa_warm_mapped"
export ds_name="ljs_ipa_v2"
export speakers="$ds_name,all"
export batch_size=26

## GCP
# For usage with a t4 on Google Cloud Plattform
source /home/stefan_taubert/tacotron2/configs/envs/prod-gcp.sh
export custom_training_name="ljs_ipa_warm_mapped"
export ds_name="ljs_ipa"
export speakers="$ds_name,all"
export batch_size=52

## Phil
source /home/stefan/tacotron2/configs/envs/prod-phil.sh
export custom_training_name="ljs_ipa_warm_mapped"
export ds_name="ljs_ipa"
export speakers="$ds_name,all"
export batch_size=26


# Preprocessing
python -m script_ljs_pre \
  --base_dir=$base_dir \
  --data_dir="$ljs_data" \
  --ipa \
  --ignore_arcs \
  --ds_name=$ds_name \
  --auto_dl \
  --no_debugging

# Training
export hparams="batch_size=$batch_size,iters_per_checkpoint=500,epochs=500,ignore_layers=[embedding.weight,speakers_embedding.weight]"
python -m paths \
  --base_dir=$base_dir \
  --custom_training_name=$custom_training_name \
  --no_debugging
python -m script_train \
  --base_dir=$base_dir \
  --training_dir=$custom_training_name \
  --speakers=$speakers \
  --hparams=$hparams \
  --train_size=0.9 \
  --validation_size=1.0 \
  --warm_start \
  --pretrained_path="$base_dir/thchs_ipa_warm_mapped/checkpoints/6000" \
  --pretrained_model="$base_dir/thchs_ipa_warm_mapped/checkpoints/6000" \
  --pretrained_model_symbols="$base_dir/thchs_ipa_warm_mapped/filelist/symbols.json" \
  --weight_map_mode='use_map' \
  --map="maps/weights/en_chn_v1.json" \
  --no_debugging

## Continue training
export hparams="batch_size=$batch_size,iters_per_checkpoint=500,epochs=500"
python -m script_train \
  --base_dir=$base_dir \
  --training_dir=$custom_training_name \
  --hparams=$hparams \
  --continue_training \
  --no_debugging

# Inference
python -m script_dl_waveglow_pretrained \
  --pretrained_dir=$pretrained_dir \
  --no_debugging
export text_map="maps/inference/en_v1.json"
export speaker="$ds_name,1"

export text="examples/ipa/north_sven_orig.txt"
python -m script_inference --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --lang=ipa --ignore_tones --ignore_arcs --speaker=$speaker --waveglow=$waveglow --map=$text_map --no_debugging --custom_checkpoint=''

export text="examples/chn/thchs.txt"
python -m script_inference --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --lang=chn --ignore_tones --ignore_arcs --speaker=$speaker --waveglow=$waveglow --map=$text_map --no_debugging --custom_checkpoint=''

export text="examples/ger/example.txt"
python -m script_inference --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --lang=ger --ignore_tones --ignore_arcs --speaker=$speaker --waveglow=$waveglow --map=$text_map --no_debugging --custom_checkpoint=''

export text="examples/ger/nord.txt"
python -m script_inference --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --lang=ger --ignore_tones --ignore_arcs --speaker=$speaker --waveglow=$waveglow --map=$text_map --no_debugging --custom_checkpoint=''

export text="examples/ipa/north_sven_v2.txt"
python -m script_inference --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --lang=ipa --ignore_tones --ignore_arcs --speaker=$speaker --waveglow=$waveglow --map=$text_map --no_debugging --custom_checkpoint=''

export text="examples/en/north.txt"
python -m script_inference --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --lang=en --ignore_tones --ignore_arcs --speaker=$speaker --waveglow=$waveglow --map=$text_map --no_debugging --custom_checkpoint=''

export text="examples/en/democritus_v2.txt"
python -m script_inference --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --lang=en --ignore_tones --ignore_arcs --speaker=$speaker --waveglow=$waveglow --map=$text_map --no_debugging --custom_checkpoint=''

export text="examples/ipa/north_ger.txt"
python -m script_inference --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --lang=ipa --ignore_tones --ignore_arcs --speaker=$speaker --waveglow=$waveglow --map=$text_map --no_debugging --custom_checkpoint=''
