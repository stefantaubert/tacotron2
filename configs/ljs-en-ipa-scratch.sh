########################################################################################
# LJSpeech based IPA Synthesis
########################################################################################

# Init
## Capslock
conda activate test1
export base_dir="/datasets/models/taco2pt_v2"
export datasets_dir="/datasets"
export pretrained_dir="/datasets/models/pretrained"
export custom_training_name="ljs_ipa_ms_from_scratch"
export waveglow="$pretrained_dir/waveglow_256channels_universal_v5.pt"
export ds_name="ljs_ipa_v2"
export speakers="$ds_name,all"
export batch_size=26

## GCP
# For usage with a t4 on Google Cloud Plattform
cd tacotron2
source activate taco2pytorch
export base_dir="/home/stefan_taubert/taco2pt_v2"
export datasets_dir="/home/stefan_taubert/datasets"
export pretrained_dir="$base_dir/pretrained"
export custom_training_name="ljs_ipa_ms_from_scratch"
export waveglow="$pretrained_dir/waveglow_256channels_universal_v5.pt"
export ds_name="ljs_ipa"
export speakers="$ds_name,all"
export batch_size=52

## Phil
cd tacotron2
conda activate test2
export base_dir="/home/stefan/taco2pt_v2"
export datasets_dir="/home/stefan/datasets"
export pretrained_dir="$base_dir/pretrained"
export custom_training_name="ljs_ipa_ms_from_scratch"
export waveglow="$pretrained_dir/waveglow_256channels_universal_v5.pt"
export ds_name="ljs_ipa"
export speakers="$ds_name,all"
export batch_size=26


# Preprocessing
python script_ljs_pre.py \
  --base_dir=$base_dir \
  --data_dir="$datasets_dir/LJSpeech-1.1" \
  --ipa \
  --ignore_arcs \
  --ds_name=$ds_name \
  --auto_dl \
  --no_debugging

# Training from scratch
export hparams="batch_size=$batch_size,iters_per_checkpoint=500,epochs=500"
python paths.py --base_dir=$base_dir --custom_training_name=$custom_training_name --no_debugging
python script_train.py --base_dir=$base_dir --training_dir=$custom_training_name --speakers=$speakers --hparams=$hparams --no_debugging
#python -m script_train --base_dir=$base_dir --training_dir=$custom_training_name --speakers=$speakers --hparams=$hparams --no_debugging

## Continue training
export hparams="batch_size=$batch_size,iters_per_checkpoint=500,epochs=500"
python script_train.py --base_dir=$base_dir --training_dir=$custom_training_name --hparams=$hparams --continue_training --no_debugging

# Inference
python script_dl_waveglow_pretrained.py --pretrained_dir=$pretrained_dir --no_debugging
export text_map="maps/inference/en_v1.json"
export speaker="ljs_ipa,1"

export text="examples/ipa/north_sven_orig.txt"
python script_inference.py --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --lang=ipa --ignore_tones --ignore_arcs  --speaker=$speaker --waveglow=$waveglow --custom_checkpoint=81500 --map=$text_map --no_debugging

export text="examples/chn/thchs.txt"
python script_inference.py --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --lang=chn --ignore_tones --ignore_arcs  --speaker=$speaker --waveglow=$waveglow --map=$text_map --no_debugging

export text="examples/ger/example.txt"
python script_inference.py --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --lang=ger --ignore_tones --ignore_arcs --speaker=$speaker --waveglow=$waveglow --custom_checkpoint=79000 --map=$text_map --no_debugging

export text="examples/ger/nord.txt"
python script_inference.py --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --lang=ger --ignore_tones --ignore_arcs --speaker=$speaker --waveglow=$waveglow --custom_checkpoint=79000 --map=$text_map --no_debugging

export text="examples/ipa/north_sven_v2.txt"
python script_inference.py --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --lang=ipa --ignore_tones --ignore_arcs --speaker=$speaker --waveglow=$waveglow --map=$text_map --no_debugging

export text="examples/en/north.txt"
python script_inference.py --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --lang=en --ignore_tones --ignore_arcs --speaker=$speaker --waveglow=$waveglow --custom_checkpoint=79000 --map=$text_map --no_debugging

export text="examples/en/democritus_v2.txt"
python script_inference.py --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --lang=en --ignore_tones --ignore_arcs --speaker=$speaker --waveglow=$waveglow --custom_checkpoint=79000 --map=$text_map --no_debugging

export text="examples/ipa/north_ger.txt"
python script_inference.py --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --lang=ipa --ignore_tones --ignore_arcs --speaker=$speaker --waveglow=$waveglow --custom_checkpoint=79000 --map=$text_map --no_debugging

# Validate checkpoints
export hparams="batch_size=52"
export select_pattern=10000
python script_eval_checkpoints.py --base_dir=$base_dir --training_dir=$custom_training_name --hparams=$hparams --select=2000 --min=70000 --no_debugging

# Plot Embeddings
python plot_embeddings.py --base_dir=$base_dir --training_dir=$custom_training_name --no_debugging

# Create Inference Map
python script_create_map_template.py \
  --a="$base_dir/ljs_ipa_ms_from_scratch/filelist/symbols.json" \
  --b="examples/ipa/corpora.txt" \
  --mode="infer"
  --ignore_tones
  --ignore_arcs
  --no_debugging

# Update Inference Map
python script_create_map_template.py \
  --a="$base_dir/ljs_ipa_ms_from_scratch/filelist/symbols.json" \
  --b="examples/ipa/corpora.txt" \
  --existing_map="maps/inference/en_v1.json"
  --mode="infer"
  --ignore_tones
  --ignore_arcs
  --no_debugging


# Validate
python script_dl_waveglow_pretrained.py --pretrained_dir=$pretrained --no_debugging

export utterance="random-val"
python script_validate.py --no_debugging --base_dir=$base_dir --training_dir=$custom_training_name --waveglow=$waveglow --utterance=$utterance --custom_checkpoint=113204
export utterance="LJ002-0205"
python script_validate.py --no_debugging --base_dir=$base_dir --training_dir=$custom_training_name --waveglow=$waveglow --utterance=$utterance --custom_checkpoint=113204
export utterance="LJ006-0229"
python script_validate.py --no_debugging --base_dir=$base_dir --training_dir=$custom_training_name --waveglow=$waveglow --utterance=$utterance --custom_checkpoint=113204
export utterance="LJ027-0076"
python script_validate.py --no_debugging --base_dir=$base_dir --training_dir=$custom_training_name --waveglow=$waveglow --utterance=$utterance --custom_checkpoint=113204
