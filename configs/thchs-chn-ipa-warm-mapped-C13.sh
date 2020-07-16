########################################################################################
# THCHS-30 based IPA Synthesis with Chinese Accents
########################################################################################
# Config: toneless, without arcs
# you have to first train ljs-en-ipa

# Init

## Capslock Dev
source /datasets/code/tacotron2-dev/configs/envs/dev-caps.sh
export custom_training_name="thchs_ipa_warm_mapped_C13"
export ds_name="thchs_v5"
export speakers="$ds_name,C13"
export batch_size=?

## Capslock GCP
source /datasets/code/tacotron2-dev/configs/envs/prod-caps.sh
export custom_training_name="thchs_ipa_warm_mapped_C13"
export ds_name="thchs_v5"
export speakers="$ds_name,C13"
export batch_size=10

## GCP
# For usage with a t4 on Google Cloud Plattform
source /home/stefan_taubert/tacotron2/configs/envs/prod-gcp.sh
export custom_training_name="thchs_ipa_warm_mapped_C13"
export ds_name="thchs_v5"
export speakers="$ds_name,C13"
export batch_size=45

## Phil
source /home/stefan/tacotron2/configs/envs/prod-phil.sh
export custom_training_name="thchs_ipa_warm_mapped_C13"
export ds_name="thchs_v5"
export speakers="$ds_name,C13"
export batch_size=?


# Preprocessing
python script_thchs_pre.py \
  --base_dir=$base_dir \
  --data_dir="$datasets_dir/thchs" \
  --data_conversion_dir="$datasets_dir/thchs_16bit_22050kHz" \
  --ignore_arcs \
  --ignore_tones \
  --auto_dl \
  --auto_convert \
  --ds_name=$ds_name \
  --no_debugging

# Training
export hparams="batch_size=$batch_size,iters_per_checkpoint=500,epochs=2000,ignore_layers=[embedding.weight,speakers_embedding.weight]"
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
  --pretrained_path="$base_dir/ljs_ipa_ms_from_scratch/checkpoints/79000" \
  --pretrained_model="$base_dir/ljs_ipa_ms_from_scratch/checkpoints/79000" \
  --pretrained_model_symbols="$base_dir/ljs_ipa_ms_from_scratch/filelist/symbols.json" \
  --weight_map_mode='use_map' \
  --map="maps/weights/chn_en_v1.json" \
  --no_debugging

## Continue training
export hparams="batch_size=$batch_size,iters_per_checkpoint=500,epochs=2000"
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
export text_map="maps/inference/chn_v1.json"
export speaker="$ds_name,C13"

export text="examples/ipa/north_sven_orig.txt"
python -m script_inference --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --lang=ipa --ignore_tones --ignore_arcs --speaker=$speaker --waveglow=$waveglow --map=$text_map --no_debugging --custom_checkpoint=''

export text="examples/chn/thchs.txt"
python -m script_inference --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --lang=chn --ignore_tones --ignore_arcs --speaker=$speaker --waveglow=$waveglow --map=$text_map --no_debugging --custom_checkpoint=''

export text="examples/ger/example.txt"
python -m script_inference --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --lang=ger --ignore_tones --ignore_arcs --speaker=$speaker --waveglow=$waveglow --custom_checkpoint='' --map=$text_map --no_debugging

export text="examples/ger/nord.txt"
python -m script_inference --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --lang=ger --ignore_tones --ignore_arcs --speaker=$speaker --waveglow=$waveglow --custom_checkpoint=''--map=$text_map --no_debugging

export text="examples/ipa/north_sven_v2.txt"
python -m script_inference --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --lang=ipa --ignore_tones --ignore_arcs --speaker=$speaker --waveglow=$waveglow --map=$text_map --no_debugging --custom_checkpoint=''

export text="examples/en/north.txt"
python -m script_inference --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --lang=en --ignore_tones --ignore_arcs --speaker=$speaker --waveglow=$waveglow --map=$text_map --no_debugging --custom_checkpoint=''

export text="examples/en/democritus_v2.txt"
python -m script_inference --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --lang=en --ignore_tones --ignore_arcs --speaker=$speaker --waveglow=$waveglow --map=$text_map --no_debugging --custom_checkpoint=''

export text="examples/ipa/north_ger.txt"
python -m script_inference --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --lang=ipa --ignore_tones --ignore_arcs --speaker=$speaker --waveglow=$waveglow --map=$text_map --no_debugging --custom_checkpoint=''
