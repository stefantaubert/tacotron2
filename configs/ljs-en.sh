########################################################################################
# LJSpeech based Synthesis of English Characters
########################################################################################
# old and not tested

export base_dir="/datasets/models/taco2pt_v2"

# Preprocessing
export ljs_dir="/datasets/LJSpeech-1.1"
export ds_name="ljs_en_v2"
python script_ljs_pre.py --base_dir=$base_dir --data_dir=$ljs_dir --ds_name=$ds_name --no_debugging

# Training
export custom_training_name="debug_en"
export hparams="batch_size=26,iters_per_checkpoint=500"

## From scratch
python paths.py --base_dir=$base_dir --custom_training_name=$custom_training_name --no_debugging
python script_train.py --base_dir=$base_dir --training_dir=$custom_training_name --ds_name=$ds_name --speaker='1' --hparams=$hparams --no_debugging

## Using pretrained model
export pretrained_path="/datasets/models/pretrained/tacotron2_statedict.pt"
python paths.py --base_dir=$base_dir --custom_training_name=$custom_training_name --no_debugging
python script_train.py --base_dir=$base_dir --training_dir=$custom_training_name --ds_name=$ds_name --speaker='1' --hparams=$hparams --warm_start --pretrained_path=$pretrained_path --no_debugging

## Continue training
python script_train.py --base_dir=$base_dir --training_dir=$custom_training_name --hparams=$hparams --continue_training --no_debugging

# Inference
export waveglow="/datasets/models/pretrained/waveglow_256channels_universal_v5.pt"
export text="examples/ipa/north_sven_v2.txt"
python script_inference.py --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --lang=ipa --ignore_tones --ignore_arcs --waveglow=$waveglow --no_debugging
export text="examples/en/north.txt"
python script_inference.py --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --ignore_tones --ignore_arcs --waveglow=$waveglow --no_debugging