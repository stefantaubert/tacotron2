########################################################################################
# LJSpeech based IPA Synthesis
########################################################################################
# for running on Google Cloud T4 image

export base_dir="/home/stefan_taubert/taco2pt_v2"

screen / screen -r
source activate taco2pytorch
cd tacotron2

# Preprocessing
export ljs_dir="/home/stefan_taubert/datasets/LJSpeech-1.1"
export ds_name="ljs"
python script_ljs_pre.py --base_dir=$base_dir --data_dir=$ljs_dir --ipa --ignore_arcs --ds_name=$ds_name --no_debugging

# Training
export hparams="batch_size=52,iters_per_checkpoint=500,epochs=500"

## From scratch
export custom_training_name="ljs_ipa_from_scratch"
python paths.py --base_dir=$base_dir --custom_training_name=$custom_training_name --no_debugging
#export final_training_dir=$(python paths.py --base_dir=$base_dir --custom_training_name=$custom_training_name --no_debugging)
python script_train.py --base_dir=$base_dir --training_dir=$custom_training_name --ds_name=$ds_name --speaker='1' --hparams=$hparams --no_debugging

## Using pretrained model
export custom_training_name="ljs_ipa_from_pretrained"
export pretrained_path="/home/stefan_taubert/taco2pt_v2/pretrained/tacotron2_statedict.pt"
python paths.py --base_dir=$base_dir --custom_training_name=$custom_training_name --no_debugging
python script_train.py --base_dir=$base_dir --training_dir=$custom_training_name --ds_name=$ds_name --speaker='1' --hparams=$hparams --warm_start --pretrained_path=$pretrained_path --no_debugging

## Continue training
export custom_training_name="ljs_ipa_from_scratch"
python script_train.py --base_dir=$base_dir --training_dir=$custom_training_name --hparams=$hparams --continue_training --no_debugging

# Inference
export pretrained="/home/stefan_taubert/taco2pt_v2/pretrained"
python script_dl_waveglow_pretrained.py --pretrained_dir=$pretrained --no_debugging

export waveglow="/home/stefan_taubert/taco2pt_v2/pretrained/waveglow_256channels_universal_v5.pt"

export text="examples/ipa/north_sven_v2.txt"
python script_inference.py --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --is_ipa --ignore_tones --ignore_arcs --waveglow=$waveglow --no_debugging

export text="examples/en/north.txt"
python script_inference.py --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --ignore_tones --ignore_arcs --waveglow=$waveglow --no_debugging