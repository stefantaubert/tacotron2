########################################################################################
# THCHS-30 based IPA Synthesis with Chinese Accents
########################################################################################
# Config: toneless, without arcs
# you have to first train ljs-en-ipa

# Init

## Capslock Dev
source /datasets/code/tacotron2/configs/envs/dev-caps.sh
export custom_training_name="thchs_ipa_warm_mapped_all"
export ds_name="thchs_nosil"
export speakers="$ds_name,all"
export batch_size=17

## Capslock GCP
source /datasets/code/tacotron2/configs/envs/prod-caps.sh
export custom_training_name="thchs_ipa_warm_mapped_all"
export ds_name="thchs_nosil"
export speakers="$ds_name,all"
export batch_size=0

## GCP
# For usage with a t4 on Google Cloud Plattform
source /home/stefan_taubert/tacotron2/configs/envs/prod-gcp.sh
export custom_training_name="thchs_ipa_warm_mapped_all"
export ds_name="thchs_nosil"
export speakers="$ds_name,all"
export batch_size=45

## Phil
source /home/stefan/tacotron2/configs/envs/prod-phil.sh
export custom_training_name="thchs_ipa_warm_mapped_all"
export ds_name="thchs_nosil"
export speakers="$ds_name,all"
export batch_size=0


# Preprocessing

python ./src/pre/thchs/script_dl.py \
  --no_debugging \
  --data_dir=$thchs_original_data \
  --ds_name=$ds_name

python ./src/pre/thchs/script_upsample.py \
  --no_debugging \
  --data_src_dir=$thchs_original_data \
  --data_dest_dir=$thchs_upsampled_data

python ./src/pre/thchs/script_remove_silence.py \
  --no_debugging \
  --data_src_dir=$thchs_upsampled_data \
  --data_dest_dir=$thchs_nosil_data \
  --chunk_size=5 \
  --threshold_start=-25 \
  --threshold_end=-35 \
  --buffer_start_ms=100 \
  --buffer_end_ms=150

python ./src/pre/thchs/script_pre.py \
  --no_debugging \
  --base_dir=$base_dir \
  --data_dir="$thchs_nosil_data" \
  --ignore_arcs \
  --ignore_tones \
  --ds_name=$ds_name

# Training
export hparams="batch_size=$batch_size,iters_per_checkpoint=500,epochs_per_checkpoint=1,epochs=2000,ignore_layers=[embedding.weight,speakers_embedding.weight]"
python ./src/script_paths.py \
  --base_dir=$base_dir \
  --custom_training_name=$custom_training_name \
  --no_debugging
python ./src/tacotron/script_train.py \
  --base_dir=$base_dir \
  --training_dir=$custom_training_name \
  --speakers=$speakers \
  --hparams=$hparams \
  --seed=1234 \
  --validation_size=0.1 \
  --test_size=0.01 \
  --warm_start \
  --pretrained_path="/datasets/gcp_home/ljs_ipa_ms_from_scratch/checkpoints/113500" \
  --pretrained_model="/datasets/gcp_home/ljs_ipa_ms_from_scratch/checkpoints/113500" \
  --pretrained_model_symbols="/datasets/gcp_home/ljs_ipa_ms_from_scratch/filelist/symbols.json" \
  --weight_map_mode='use_map' \
  --map="maps/weights/chn_en_v1.json" \
  --no_debugging

## Continue training
export hparams="batch_size=$batch_size,iters_per_checkpoint=500,epochs_per_checkpoint=1,epochs=2000"
python ./src/tacotron/script_train.py \
  --base_dir=$base_dir \
  --training_dir=$custom_training_name \
  --hparams=$hparams \
  --continue_training \
  --no_debugging

# Inference
python ./src/waveglow/script_dl_pretrained.py \
  --destination=$waveglow \
  --auto_convert \
  --no_debugging

export text_map="maps/inference/chn_v1.json"
export speaker="$ds_name,D31"

# epoch 19 - 13019

export text="examples/ipa/north_sven_orig.txt"
python ./src/tacotron/script_inference.py --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --lang=ipa --ignore_tones --ignore_arcs --speaker=$speaker --waveglow=$waveglow --map=$text_map --no_debugging --analysis --custom_checkpoint='13019'

export text="examples/ger/nord.txt"
python ./src/tacotron/script_inference.py --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --lang=ger --ignore_tones --ignore_arcs --speaker=$speaker --waveglow=$waveglow --map=$text_map --no_debugging --analysis --custom_checkpoint='13019'

export text="examples/en/north.txt"
python ./src/tacotron/script_inference.py --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --lang=en --ignore_tones --ignore_arcs --speaker=$speaker --waveglow=$waveglow --map=$text_map --no_debugging --analysis --custom_checkpoint='13019'

export text="examples/ipa/north_ger.txt"
python ./src/tacotron/script_inference.py --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --lang=ipa --ignore_tones --ignore_arcs --speaker=$speaker --waveglow=$waveglow --map=$text_map --no_debugging --analysis --custom_checkpoint='13019'

# Validate
python ./src/waveglow/script_dl_pretrained.py \
  --destination=$waveglow \
  --auto_convert \
  --no_debugging
  
export utterance="random-val"
python ./src/tacotron/script_validate.py --no_debugging --base_dir=$base_dir --training_dir=$custom_training_name --waveglow=$waveglow --utterance=$utterance --custom_checkpoint='13019'

export utterance="D31_832"
python ./src/tacotron/script_validate.py --no_debugging --base_dir=$base_dir --training_dir=$custom_training_name --waveglow=$waveglow --utterance=$utterance --custom_checkpoint='13019'

export utterance="D31_764"
python ./src/tacotron/script_validate.py --no_debugging --base_dir=$base_dir --training_dir=$custom_training_name --waveglow=$waveglow --utterance=$utterance --custom_checkpoint='13019'

export utterance="D31_917"
python ./src/tacotron/script_validate.py --no_debugging --base_dir=$base_dir --training_dir=$custom_training_name --waveglow=$waveglow --utterance=$utterance --custom_checkpoint='13019'

export utterance="D31_769"
python ./src/tacotron/script_validate.py --no_debugging --base_dir=$base_dir --training_dir=$custom_training_name --waveglow=$waveglow --utterance=$utterance --custom_checkpoint='13019'

export utterance="D31_953"
python ./src/tacotron/script_validate.py --no_debugging --base_dir=$base_dir --training_dir=$custom_training_name --waveglow=$waveglow --utterance=$utterance --custom_checkpoint='13019'

# D31_832
# D31_764
# D31_917
# D31_769
# D31_953
# D31_893
# D31_932
# D31_798
# D31_860
# D31_925
# D31_774
# D31_782
# D31_874
# D31_830
# D31_948
# D31_906
# D31_781
# D31_778
# D31_767
# D31_840
# D31_986
# D31_851
# D31_901
# D31_835
# D31_887
