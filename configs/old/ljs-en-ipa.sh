########################################################################################
# LJSpeech based IPA Synthesis
########################################################################################
## not tested

export base_dir="/datasets/models/taco2pt_v2"

# Preprocessing
export ljs_dir="/datasets/LJSpeech-1.1"
export ds_name="ljs_ipa_v2"
python script_ljs_pre.py --base_dir=$base_dir --data_dir=$ljs_dir --ipa --ignore_arcs --ds_name=$ds_name --no_debugging

# Training
export custom_training_name="debug"
export hparams="batch_size=26,iters_per_checkpoint=500"
export speakers='ljs_ipa_v2,1'

## From scratch
python paths.py --base_dir=$base_dir --custom_training_name=$custom_training_name --no_debugging
python script_train.py --base_dir=$base_dir --training_dir=$custom_training_name --ds_name=$ds_name --speakers=$speakers --hparams=$hparams --no_debugging

## Using pretrained model weights
export pretrained_path="/datasets/models/taco2pt_v2/ljs_ipa_baseline/checkpoints/49000"
export pretrained_model_symbols="/datasets/models/taco2pt_v2/ljs_ipa_baseline/filelist/symbols.json"
python paths.py --base_dir=$base_dir --custom_training_name=$custom_training_name --no_debugging
python script_train.py --base_dir=$base_dir --training_dir=$custom_training_name --ds_name=$ds_name --speakers=$speakers --hparams=$hparams --weight_map_mode='same_symbols_only' --pretrained_model=$pretrained_path --pretrained_model_symbols=$pretrained_model_symbols  --no_debugging

## Using pretrained model
export pretrained_path="/datasets/models/pretrained/tacotron2_statedict.pt"
python paths.py --base_dir=$base_dir --custom_training_name=$custom_training_name --no_debugging
python script_train.py --base_dir=$base_dir --training_dir=$custom_training_name --ds_name=$ds_name --speakers=$speakers --hparams=$hparams --warm_start --pretrained_path=$pretrained_path --no_debugging

## Continue training
python script_train.py --base_dir=$base_dir --training_dir=$custom_training_name --hparams=$hparams --continue_training --no_debugging

# Inference
export waveglow="/datasets/models/pretrained/waveglow_256channels_universal_v5.pt"
export speaker='ljs_ipa_v2,1'

export text="examples/ipa/north_sven_v2.txt"
python script_inference.py --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --lang=ipa --ignore_tones --ignore_arcs --speakers=$speakers --speaker=$speaker --waveglow=$waveglow --no_debugging
export text="examples/en/north.txt"
python script_inference.py --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --ignore_tones --ignore_arcs --speakers=$speakers --speaker=$speaker --waveglow=$waveglow --no_debugging