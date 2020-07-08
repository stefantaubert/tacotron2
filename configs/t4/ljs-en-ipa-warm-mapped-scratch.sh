########################################################################################
# LJSpeech based IPA Synthesis
########################################################################################
# For usage with a t4 on Google Cloud Plattform

# Init
screen -r
cd tacotron2
source activate taco2pytorch

# Preprocessing
export base_dir="/home/stefan_taubert/taco2pt_v2"
export ljs_dir="/home/stefan_taubert/datasets/LJSpeech-1.1"
export ds_name="ljs_ipa"
python script_ljs_pre.py --base_dir=$base_dir --data_dir=$ljs_dir --ipa --ignore_arcs --ds_name=$ds_name --no_debugging

# Training
python paths.py \
  --base_dir="/home/stefan_taubert/taco2pt_v2" \
  --custom_training_name="ljs_ipa_warm_mapped_scratch" \
  --no_debugging
python script_train.py \
  --base_dir="/home/stefan_taubert/taco2pt_v2" \
  --training_dir="ljs_ipa_warm_mapped_scratch" \
  --speakers="ljs_ipa,1" \
  --hparams="batch_size=52,iters_per_checkpoint=100,epochs=500,ignore_layers=[embedding.weight,speakers_embedding.weight]" \
  --pretrained_model="/home/stefan_taubert/taco2pt_v2/thchs_ipa_scratch/checkpoints/29000" \
  --pretrained_model_symbols="/home/stefan_taubert/taco2pt_v2/thchs_ipa_scratch/filelist/symbols.json" \
  --weight_map_mode='use_map' \
  --map="maps/weights/en_chn_v1.json" \
  --warm_start \
  --pretrained_path="/home/stefan_taubert/taco2pt_v2/thchs_ipa_scratch/checkpoints/29000" \
  --no_debugging

## Continue training
python script_train.py \
  --base_dir="/home/stefan_taubert/taco2pt_v2" \
  --training_dir="ljs_ipa_warm_mapped_scratch" \
  --hparams="batch_size=52,iters_per_checkpoint=500,epochs=500" \
  --speakers="ljs_ipa,1" \
  --continue_training \
  --no_debugging

# Inference
export base_dir="/home/stefan_taubert/taco2pt_v2"
export pretrained="/home/stefan_taubert/taco2pt_v2/pretrained"
python script_dl_waveglow_pretrained.py --pretrained_dir=$pretrained --no_debugging
export custom_training_name="ljs_ipa_warm_mapped_scratch"
export waveglow="/home/stefan_taubert/taco2pt_v2/pretrained/waveglow_256channels_universal_v5.pt"
export speakers="ljs_ipa,1"
export speaker="ljs_ipa,1"
export text_map="maps/inference/en_v1.json"

export text="examples/ipa/north_sven_v2.txt"
python script_inference.py --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --is_ipa --ignore_tones --ignore_arcs --speakers=$speakers --speaker=$speaker --waveglow=$waveglow --map=$text_map --no_debugging

export text="examples/ipa/north_sven_orig.txt"
python script_inference.py --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --is_ipa --ignore_tones --ignore_arcs --speakers=$speakers --speaker=$speaker --waveglow=$waveglow --custom_checkpoint=79000 --map=$text_map --no_debugging

export text="examples/en/north.txt"
python script_inference.py --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --ignore_tones --ignore_arcs --speakers=$speakers --speaker=$speaker --waveglow=$waveglow --custom_checkpoint=79000 --map=$text_map --no_debugging

export text="examples/en/democritus_v2.txt"
python script_inference.py --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --ignore_tones --ignore_arcs --speakers=$speakers --speaker=$speaker --waveglow=$waveglow --custom_checkpoint=79000 --map=$text_map --no_debugging

export text="examples/ipa/north_ger.txt"
python script_inference.py --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --is_ipa --ignore_tones --ignore_arcs --speakers=$speakers --speaker=$speaker --waveglow=$waveglow --custom_checkpoint=79000 --map=$text_map --no_debugging

# Validate
export base_dir="/home/stefan_taubert/taco2pt_v2"
export hparams="batch_size=52"
export speakers="ljs_ipa,1"
export custom_training_name="ljs_ipa_warm_mapped_scratch"
export select_pattern=10000
python script_eval_checkpoints.py --base_dir=$base_dir --training_dir=$custom_training_name --speakers=$speakers --hparams=$hparams --select=2000 --min=70000 --no_debugging

# Plot Embeddings
export base_dir="/home/stefan_taubert/taco2pt_v2"
export custom_training_name="ljs_ipa_warm_mapped_scratch"
python plot_embeddings.py --base_dir=$base_dir --training_dir=$custom_training_name --no_debugging

# Create Inference Map
export model_symbols="/home/stefan_taubert/taco2pt_v2/ljs_ipa_warm_mapped_scratch/filelist/symbols.json"
export corpora="examples/ipa/corpora.txt"
python script_create_map_template.py --a=$model_symbols --b=$corpora --no_debugging