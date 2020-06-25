########################################################################################
# LJSpeech based IPA Synthesis
########################################################################################
export base_dir="/home/stefan_taubert/taco2pt_v2"

# Preprocessing
export ljs_dir="/home/stefan_taubert/datasets/LJSpeech-1.1"
export ds_name="ljs"
python script_ljs_pre.py --base_dir=$base_dir --data_dir=$ljs_dir --ipa --ignore_arcs --ds_name=$ds_name --no_debugging

# Training
export hparams="batch_size=52,iters_per_checkpoint=500,epochs=500"
export speakers="ljs,1"

## From scratch
export custom_training_name="ljs_ipa_ms_from_scratch"
python paths.py --base_dir=$base_dir --custom_training_name=$custom_training_name --no_debugging
python script_train.py --base_dir=$base_dir --training_dir=$custom_training_name --ds_name=$ds_name --speakers=$speakers --hparams=$hparams --no_debugging

## Continue training
python script_train.py --base_dir=$base_dir --training_dir=$custom_training_name --hparams=$hparams --speakers=$speakers --continue_training --no_debugging

# Inference
export pretrained="/home/stefan_taubert/taco2pt_v2/pretrained"
python script_dl_waveglow_pretrained.py --pretrained_dir=$pretrained --no_debugging

export waveglow="/home/stefan_taubert/taco2pt_v2/pretrained/waveglow_256channels_universal_v5.pt"
export speaker="ljs,1"

export text="examples/ipa/north_sven_v2.txt"
python script_inference.py --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --is_ipa --ignore_tones --ignore_arcs --speakers=$speakers --speaker=$speaker --waveglow=$waveglow --no_debugging
export text="examples/en/north.txt"
python script_inference.py --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --ignore_tones --ignore_arcs --speakers=$speakers --speaker=$speaker --waveglow=$waveglow --no_debugging