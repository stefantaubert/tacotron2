########################################################################################
# THCHS-30 based IPA Synthesis
########################################################################################
# old and not tested
# Config: toneless, without arcs, no mapping

export base_dir="/datasets/models/taco2pt_v2"

# Preprocessing
export thchs_orig_dir="/datasets/thchs"
export thchs_dir="/datasets/thchs_16bit_22050kHz"
export ds_name="thchs_v5"
python script_upsample_thchs.py --data_src_dir=$thchs_orig_dir --data_dest_dir=$thchs_dir --no_debugging
python script_thchs_pre.py --base_dir=$base_dir --data_dir=$ljs_dir --ignore_arcs --ignore_tones --ds_name=$ds_name --no_debugging

# Training
export custom_training_name="debug_thchs_ipa"
export hparams="batch_size=26,iters_per_checkpoint=500,epochs=2000"
export speaker="A11"

## From scratch
python paths.py --base_dir=$base_dir --custom_training_name=$custom_training_name --no_debugging
python script_train.py --base_dir=$base_dir --training_dir=$custom_training_name --ds_name=$ds_name --speaker=$speaker --hparams=$hparams --no_debugging

## Using pretrained model
export pretrained_path="/datasets/models/pretrained/tacotron2_statedict.pt"
python paths.py --base_dir=$base_dir --custom_training_name=$custom_training_name --no_debugging
python script_train.py --base_dir=$base_dir --training_dir=$custom_training_name --ds_name=$ds_name --speaker=$speaker --hparams=$hparams --warm_start --pretrained_path=$pretrained_path --no_debugging

## Continue training
python script_train.py --base_dir=$base_dir --training_dir=$custom_training_name --hparams=$hparams --continue_training --no_debugging

# Inference
export waveglow="/datasets/models/pretrained/waveglow_256channels_universal_v5.pt"
export text="examples/ipa/north_sven_v2.txt"
python script_inference.py --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --is_ipa --ignore_tones --ignore_arcs --waveglow=$waveglow --no_debugging
export text="examples/en/north.txt"
python script_inference.py --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --ignore_tones --ignore_arcs --waveglow=$waveglow --no_debugging