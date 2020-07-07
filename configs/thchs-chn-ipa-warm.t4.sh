########################################################################################
# THCHS-30 based IPA Synthesis
########################################################################################
# old and not tested
# Config: toneless, without arcs, no mapping

# Init
screen -r
cd tacotron2
source activate taco2pytorch

# Preprocessing
export base_dir="/datasets/models/taco2pt_v2"
export thchs_orig_dir="/home/stefan_taubert/datasets/thchs"
export thchs_dir="/home/stefan_taubert/datasets/thchs_16bit_22050kHz"
export ds_name="thchs_v5"
python script_thchs_pre.py --base_dir=$base_dir --data_dir=$thchs_orig_dir --data_conversion_dir=$thchs_dir --ignore_arcs --ignore_tones --auto_dl --auto_convert --ds_name=$ds_name --no_debugging

# Training
export base_dir="/home/stefan_taubert/taco2pt_v2"
export custom_training_name="thchs_ipa_warm"
export speakers="thchs_v5,B8"
export hparams="batch_size=45,iters_per_checkpoint=500,epochs=2000,ignore_layers=[embedding.weight,speakers_embedding.weight]"
export pretrained="/home/stefan_taubert/taco2pt_v2/ljs_ipa_ms_from_scratch/checkpoints/79000"
python paths.py --base_dir=$base_dir --custom_training_name=$custom_training_name --no_debugging
python script_train.py --base_dir=$base_dir --training_dir=$custom_training_name --speakers=$speakers --hparams=$hparams --warm_start --pretrained_path=$pretrained --no_debugging

## Continue training
export base_dir="/home/stefan_taubert/taco2pt_v2"
export custom_training_name="thchs_ipa_warm"
export hparams="batch_size=45,iters_per_checkpoint=500,epochs=2000"
export speakers="thchs_v5,B8"
python script_train.py --base_dir=$base_dir --training_dir=$custom_training_name --hparams=$hparams --continue_training --speakers=$speakers --no_debugging

# Inference
export base_dir="/home/stefan_taubert/taco2pt_v2"
export custom_training_name="thchs_ipa_warm"
export pretrained="/home/stefan_taubert/taco2pt_v2/pretrained"
python script_dl_waveglow_pretrained.py --pretrained_dir=$pretrained --no_debugging
export waveglow="/home/stefan_taubert/taco2pt_v2/pretrained/waveglow_256channels_universal_v5.pt"
export speakers="thchs_v5,B8"
export speaker="thchs_v5,B8"
export text_map="maps/inference/chn_v1.json"

export text="examples/ipa/thchs.txt"
python script_inference.py --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --is_ipa --map=$text_map --ignore_tones --ignore_arcs --waveglow=$waveglow --speakers=$speakers --speaker=$speaker --no_debugging

export text="examples/ipa/north_sven_v2.txt"
python script_inference.py --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --is_ipa --map=$text_map --ignore_tones --ignore_arcs --speakers=$speakers --speaker=$speaker --waveglow=$waveglow --no_debugging

export text="examples/ipa/north_ger.txt"
python script_inference.py --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --is_ipa --map=$text_map --ignore_tones --ignore_arcs --speakers=$speakers --speaker=$speaker --waveglow=$waveglow --no_debugging

# Create Inference Map Template
export model_symbols="/home/stefan_taubert/taco2pt_v2/thchs_ipa_warm/filelist/symbols.json"
export corpora="examples/ipa/corpora.txt"
python script_create_map_template.py --mode=infer --a=$model_symbols --b=$corpora --no_debugging