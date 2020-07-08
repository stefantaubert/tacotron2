########################################################################################
# THCHS-30 based IPA Synthesis with Chinese Accents
########################################################################################
# Config: toneless, without arcs
# you have to first train ljs-en-ipa

# Init
screen -r
cd tacotron2
source activate taco2pytorch

# Preprocessing
export base_dir="/home/stefan_taubert/taco2pt_v2"
export thchs_orig_dir="/home/stefan_taubert/datasets/thchs"
export thchs_dir="/home/stefan_taubert/datasets/thchs_16bit_22050kHz"
export ds_name="thchs_v5"
python script_thchs_pre.py --base_dir=$base_dir --data_dir=$thchs_orig_dir --data_conversion_dir=$thchs_dir --ignore_arcs --ignore_tones --auto_dl --auto_convert --ds_name=$ds_name --no_debugging

# Training
# export speaker="thchs_v5,B2;thchs_v5,A2"
export base_dir="/home/stefan_taubert/taco2pt_v2"
export custom_training_name="thchs_ipa_warm_mapped"
export speakers="thchs_v5,B8"
export hparams="batch_size=45,iters_per_checkpoint=500,epochs=2000,ignore_layers=[embedding.weight,speakers_embedding.weight]"
# export speakers="thchs_v5,B8;thchs_v5,B2;thchs_v5,A2"
# export hparams="batch_size=35,iters_per_checkpoint=500,epochs=2000,ignore_layers=[embedding.weight,speakers_embedding.weight]"
export model_with_weights="/home/stefan_taubert/taco2pt_v2/ljs_ipa_ms_from_scratch/checkpoints/79000"
export model_with_weights_symbols="/home/stefan_taubert/taco2pt_v2/ljs_ipa_ms_from_scratch/filelist/symbols.json"
export map="maps/weights/chn_en_v1.json"
python paths.py --base_dir=$base_dir --custom_training_name=$custom_training_name --no_debugging
python script_train.py --base_dir=$base_dir --training_dir=$custom_training_name --speakers=$speakers --hparams=$hparams --pretrained_model=$model_with_weights --pretrained_model_symbols=$model_with_weights_symbols --weight_map_mode='use_map' --map=$map --warm_start --pretrained_path=$model_with_weights --no_debugging

## Continue training
export base_dir="/home/stefan_taubert/taco2pt_v2"
export custom_training_name="thchs_ipa_warm_mapped"
export hparams="batch_size=35,iters_per_checkpoint=500,epochs=2000"
export speakers="thchs_v5,B8"
python script_train.py --base_dir=$base_dir --training_dir=$custom_training_name --hparams=$hparams --continue_training --speakers=$speakers --no_debugging

# Inference
export base_dir="/home/stefan_taubert/taco2pt_v2"
export custom_training_name="thchs_ipa_warm_mapped"
export pretrained="/home/stefan_taubert/taco2pt_v2/pretrained"
python script_dl_waveglow_pretrained.py --pretrained_dir=$pretrained --no_debugging
export waveglow="/home/stefan_taubert/taco2pt_v2/pretrained/waveglow_256channels_universal_v5.pt"
export text_map="maps/inference/chn_v1.json"
export speakers="thchs_v5,B8"
export speaker="thchs_v5,B8"

export text="examples/ipa/thchs.txt"
python script_inference.py --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --lang=ipa --map=$text_map --ignore_tones --ignore_arcs --speakers=$speakers --speaker=$speaker --waveglow=$waveglow --custom_checkpoint=500 --no_debugging

export text="examples/ipa/north_sven_v2.txt"
python script_inference.py --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --lang=ipa --ignore_tones --ignore_arcs --waveglow=$waveglow --map=$text_map --speakers=$speakers --speaker=$speaker --custom_checkpoint=500 --no_debugging

export text="examples/en/north.txt"
python script_inference.py --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --lang=en --ignore_tones --ignore_arcs --waveglow=$waveglow --map=$text_map --speakers=$speakers --speaker=$speaker --custom_checkpoint=500 --no_debugging

export text="examples/en/democritus_v2.txt"
python script_inference.py --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --lang=en --ignore_tones --ignore_arcs --speakers=$speakers --speaker=$speaker --waveglow=$waveglow --map=$text_map --no_debugging

export text="examples/ipa/north_ger.txt"
python script_inference.py --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --lang=ipa --map=$text_map --ignore_tones --ignore_arcs --speakers=$speakers --speaker=$speaker --waveglow=$waveglow --custom_checkpoint=500 --no_debugging

# Create Weight Map Template
export model_symbols="/home/stefan_taubert/taco2pt_v2/ds/thchs/all_symbols.json"
export model_with_weights_symbols="/home/stefan_taubert/taco2pt_v2/ds/ljs_ipa/all_symbols.json"
python script_create_map_template.py --mode=weights --a=$model_with_weights_symbols --b=$model_symbols --no_debugging
