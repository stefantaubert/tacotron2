########################################################################################
# THCHS-30 based IPA Synthesis with Chinese Accents
########################################################################################
# Config: toneless, without arcs
# you have to first train ljs-en-ipa

# Init

## Capslock Dev
source /datasets/code/tacotron2/configs/envs/dev-caps.sh
export custom_training_name="thchs_ipa_warm_mapped_all_tones"
export ds_name="thchs_nosil_tones"
export speakers="$ds_name,all"
export batch_size=17

## Capslock GCP
source /datasets/code/tacotron2/configs/envs/prod-caps.sh
export custom_training_name="thchs_ipa_warm_mapped_all_tones"
export ds_name="thchs_nosil_tones"
export speakers="$ds_name,all"
export batch_size=0

## GCP
# For usage with a t4 on Google Cloud Plattform
source /home/stefan_taubert/tacotron2/configs/envs/prod-gcp.sh
export custom_training_name="thchs_ipa_warm_mapped_all_tones"
export ds_name="thchs_nosil_tones"
export speakers="$ds_name,all"
export batch_size=45

## Phil
source /home/stefan/tacotron2/configs/envs/prod-phil.sh
export custom_training_name="thchs_ipa_warm_mapped_all_tones"
export ds_name="thchs_nosil_tones"
export speakers="$ds_name,all"
export batch_size=0


# Preprocessing

python -m src.runner thchs-dl \
  --data_dir=$thchs_original_data \
  --ds_name=$ds_name

python -m src.runner thchs-upsample \
  --data_src_dir=$thchs_original_data \
  --data_dest_dir=$thchs_upsampled_data

python -m src.runner thchs-remove-silence \
  --data_src_dir=$thchs_upsampled_data \
  --data_dest_dir=$thchs_nosil_data \
  --chunk_size=5 \
  --threshold_start=-25 \
  --threshold_end=-35 \
  --buffer_start_ms=100 \
  --buffer_end_ms=150

python -m src.runner calc-mels \
  --base_dir=$base_dir \
  --data_dir="$thchs_nosil_data" \
  --ignore_arcs \
  --ds_name=$ds_name

python -m src.runner thchs-pre \
  --base_dir=$base_dir \
  --data_dir="$thchs_nosil_data" \
  --ignore_arcs \
  --ds_name=$ds_name

# Create Weights Map (do not run again)
python -m src.runner create-map \
  --a="$base_dir/ds/ljs_ipa_v2/all_symbols.json" \
  --b="$base_dir/ds/thchs_nosil_tones/all_symbols.json" \
  --out="maps/weights/chn_en_tones.json" \
  --mode="weights" \
  --ignore_arcs


# Training
export hparams="batch_size=$batch_size,iters_per_checkpoint=500,epochs_per_checkpoint=1,epochs=2000,ignore_layers=[embedding.weight,speakers_embedding.weight]"
python -m src.runner paths \
  --base_dir=$base_dir \
  --custom_training_name=$custom_training_name
python -m src.runner tacotron-train \
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
  --weights_map="maps/weights/chn_en_tones.json"

## Continue training
export hparams="batch_size=$batch_size,iters_per_checkpoint=500,epochs_per_checkpoint=1,epochs=2000"
python -m src.runner tacotron-train \
  --base_dir=$base_dir \
  --training_dir=$custom_training_name \
  --hparams=$hparams \
  --continue_training

# Inference
python -m src.runner waveglow-dl \
  --destination=$waveglow \
  --auto_convert

export text_map="maps/inference/chn_v1.json"
export speaker="$ds_name,D31"
#export custom_checkpoint='13019' # 12500_epoch-19_it-131_grad-0.781158_train-0.298055_val-0.351991_avg-0.325023
#export custom_checkpoint='50777' # 50777_epoch-77_it-650_grad-0.437439_train-0.299355_val-0.344239_avg-0.321797
#export custom_checkpoint='64020' # 64020_epoch-97_it-650_grad-1.191315_train-0.301025_val-0.341098_avg-0.321061
#export custom_checkpoint='71832' # 71832_epoch-109_it-650_grad-0.574588_train-0.266069_val-0.341585_avg-0.303827
export custom_checkpoint='72483' # 72483_epoch-110_it-650_grad-0.478423_train-0.309495_val-0.342642_avg-0.326069

export text="examples/chn/north_wiki.txt"
python -m src.runner tacotron-infer --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --lang=chn --ignore_arcs --speaker=$speaker --waveglow=$waveglow --map=$text_map --analysis --custom_checkpoint=$custom_checkpoint

export text="examples/chn/north.txt"
python -m src.runner tacotron-infer --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --lang=chn --ignore_arcs --speaker=$speaker --waveglow=$waveglow --map=$text_map --analysis --custom_checkpoint=$custom_checkpoint

export text="examples/ipa/north_sven_orig.txt"
python -m src.runner tacotron-infer --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --lang=ipa --ignore_tones --ignore_arcs --speaker=$speaker --waveglow=$waveglow --map=$text_map --analysis --custom_checkpoint=$custom_checkpoint

export text="examples/ger/nord.txt"
python -m src.runner tacotron-infer --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --lang=ger --ignore_tones --ignore_arcs --speaker=$speaker --waveglow=$waveglow --map=$text_map --analysis --custom_checkpoint=$custom_checkpoint

export text="examples/en/north.txt"
python -m src.runner tacotron-infer --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --lang=en --ignore_tones --ignore_arcs --speaker=$speaker --waveglow=$waveglow --map=$text_map --analysis --custom_checkpoint=$custom_checkpoint

export text="examples/ipa/north_ger.txt"
python -m src.runner tacotron-infer --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --lang=ipa --ignore_tones --ignore_arcs --speaker=$speaker --waveglow=$waveglow --map=$text_map --analysis --custom_checkpoint=$custom_checkpoint

# Validate
python -m src.runner waveglow-dl \
  --destination=$waveglow \
  --auto_convert
  
export utterance="random-val"
python -m src.runner tacotron-validate --base_dir=$base_dir --training_dir=$custom_training_name --waveglow=$waveglow --utterance=$utterance --custom_checkpoint=$custom_checkpoint

export utterance="D31_832"
python -m src.runner tacotron-validate --base_dir=$base_dir --training_dir=$custom_training_name --waveglow=$waveglow --utterance=$utterance --custom_checkpoint=$custom_checkpoint

export utterance="D31_764"
python -m src.runner tacotron-validate --base_dir=$base_dir --training_dir=$custom_training_name --waveglow=$waveglow --utterance=$utterance --custom_checkpoint=$custom_checkpoint

export utterance="D31_917"
python -m src.runner tacotron-validate --base_dir=$base_dir --training_dir=$custom_training_name --waveglow=$waveglow --utterance=$utterance --custom_checkpoint=$custom_checkpoint

export utterance="D31_769"
python -m src.runner tacotron-validate --base_dir=$base_dir --training_dir=$custom_training_name --waveglow=$waveglow --utterance=$utterance --custom_checkpoint=$custom_checkpoint

export utterance="D31_953"
python -m src.runner tacotron-validate --base_dir=$base_dir --training_dir=$custom_training_name --waveglow=$waveglow --utterance=$utterance --custom_checkpoint=$custom_checkpoint

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
