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

# Training from scratch
export base_dir="/home/stefan_taubert/taco2pt_v2"
export custom_training_name="thchs_ipa_scratch"
export hparams="batch_size=35,iters_per_checkpoint=500,epochs=2000"
export speakers="thchs_v5,B8;thchs_v5,B2;thchs_v5,A2"
python paths.py --base_dir=$base_dir --custom_training_name=$custom_training_name --no_debugging
python script_train.py --base_dir=$base_dir --training_dir=$custom_training_name --speakers=$speakers --hparams=$hparams --no_debugging

## Continue training
export base_dir="/home/stefan_taubert/taco2pt_v2"
export custom_training_name="thchs_ipa_scratch"
export hparams="batch_size=35,iters_per_checkpoint=500,epochs=2000"
export speakers="thchs_v5,B8;thchs_v5,B2;thchs_v5,A2"
python script_train.py --base_dir=$base_dir --training_dir=$custom_training_name --hparams=$hparams --continue_training --speakers=$speakers --no_debugging

# Inference
export base_dir="/home/stefan_taubert/taco2pt_v2"
export custom_training_name="thchs_ipa_scratch"
export pretrained="/home/stefan_taubert/taco2pt_v2/pretrained"
python script_dl_waveglow_pretrained.py --pretrained_dir=$pretrained --no_debugging
export waveglow="/home/stefan_taubert/taco2pt_v2/pretrained/waveglow_256channels_universal_v5.pt"
export speakers="thchs_v5,B8;thchs_v5,B2;thchs_v5,A2"
export speaker="thchs_v5,A2"
export text_map="maps/inference/chn_v1.json"

export text="examples/ipa/thchs.txt"
python script_inference.py --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --is_ipa --map=$text_map --ignore_tones --ignore_arcs --waveglow=$waveglow --speakers=$speakers --speaker=$speaker --no_debugging

export text="examples/ipa/north_sven_v2.txt"
python script_inference.py --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --is_ipa --map=$text_map --ignore_tones --ignore_arcs --speakers=$speakers --speaker=$speaker --waveglow=$waveglow --no_debugging

export text="examples/ipa/north_ger.txt"
python script_inference.py --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --is_ipa --map=$text_map --ignore_tones --ignore_arcs --speakers=$speakers --speaker=$speaker --waveglow=$waveglow --no_debugging

# Create Inference Map Template
export model_symbols="/home/stefan_taubert/taco2pt_v2/thchs_ipa_scratch/filelist/symbols.json"
export corpora="examples/ipa/corpora.txt"
python script_create_map_template.py --mode=infer --a=$model_symbols --b=$corpora --no_debugging