########################################################################################
# LJSpeech based IPA Synthesis
########################################################################################

# Init

## Capslock Dev
source /datasets/code/tacotron2-dev/configs/envs/dev-caps.sh
export custom_training_name="ljs_ipa_ms_from_scratch"
export ds_name="ljs_ipa_v2"
export speakers="$ds_name,all"
export batch_size=26

## Capslock GCP
source /datasets/code/tacotron2-dev/configs/envs/prod-caps.sh
export custom_training_name="ljs_ipa_ms_from_scratch"
export ds_name="ljs_ipa_v2"
export speakers="$ds_name,all"
export batch_size=26

## GCP
# For usage with a t4 on Google Cloud Plattform
source /home/stefan_taubert/tacotron2/configs/envs/prod-gcp.sh
export custom_training_name="ljs_ipa_ms_from_scratch"
export ds_name="ljs_ipa"
export speakers="$ds_name,all"
export batch_size=52

## Phil
source /home/stefan/tacotron2/configs/envs/prod-phil.sh
export custom_training_name="ljs_ipa_ms_from_scratch"
export ds_name="ljs_ipa"
export speakers="$ds_name,all"
export batch_size=26


# Preprocessing
python -m script_ljs_pre \
  --base_dir=$base_dir \
  --data_dir="$datasets_dir/LJSpeech-1.1" \
  --ipa \
  --ignore_arcs \
  --ds_name=$ds_name \
  --auto_dl \
  --no_debugging


# Training from scratch
export hparams="batch_size=$batch_size,iters_per_checkpoint=500,epochs=1000"
python -m paths --base_dir=$base_dir --custom_training_name=$custom_training_name --no_debugging
python -m script_train --base_dir=$base_dir --training_dir=$custom_training_name --speakers=$speakers --hparams=$hparams --train_size=0.9 --validation_size=1.0 --no_debugging


## Continue training
export hparams="batch_size=$batch_size,iters_per_checkpoint=500,epochs=1000"
python -m script_train --base_dir=$base_dir --training_dir=$custom_training_name --hparams=$hparams --continue_training --no_debugging


# Inference
python -m script_dl_waveglow_pretrained --pretrained_dir=$pretrained_dir --no_debugging
export text_map="maps/inference/en_v1.json"
export speaker="$ds_name,1"

export text="examples/ipa/north_sven_orig.txt"
python -m script_inference --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --lang=ipa --ignore_tones --ignore_arcs --speaker=$speaker --waveglow=$waveglow --map=$text_map --no_debugging --custom_checkpoint=''

export text="examples/chn/thchs.txt"
python -m script_inference --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --lang=chn --ignore_tones --ignore_arcs --speaker=$speaker --waveglow=$waveglow --map=$text_map --no_debugging

export text="examples/ger/example.txt"
python -m script_inference --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --lang=ger --ignore_tones --ignore_arcs --speaker=$speaker --waveglow=$waveglow --custom_checkpoint=79000 --map=$text_map --no_debugging

export text="examples/ger/nord.txt"
python -m script_inference --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --lang=ger --ignore_tones --ignore_arcs --speaker=$speaker --waveglow=$waveglow --custom_checkpoint=79000 --map=$text_map --no_debugging

export text="examples/ipa/north_sven_v2.txt"
python -m script_inference --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --lang=ipa --ignore_tones --ignore_arcs --speaker=$speaker --waveglow=$waveglow --map=$text_map --no_debugging

export text="examples/en/north.txt"
python -m script_inference --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --lang=en --ignore_tones --ignore_arcs --speaker=$speaker --waveglow=$waveglow --custom_checkpoint=79000 --map=$text_map --no_debugging

export text="examples/en/democritus_v2.txt"
python -m script_inference --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --lang=en --ignore_tones --ignore_arcs --speaker=$speaker --waveglow=$waveglow --custom_checkpoint=79000 --map=$text_map --no_debugging

export text="examples/ipa/north_ger.txt"
python -m script_inference --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --lang=ipa --ignore_tones --ignore_arcs --speaker=$speaker --waveglow=$waveglow --custom_checkpoint=79000 --map=$text_map --no_debugging

# Validate checkpoints
export hparams="batch_size=$batch_size"
export select_pattern=10000
python -m script_eval_checkpoints --base_dir=$base_dir --training_dir=$custom_training_name --hparams=$hparams --select=2000 --min=70000 --no_debugging

# Plot Embeddings
python -m plot_embeddings \
  --base_dir=$base_dir \
  --training_dir=$custom_training_name \
  --no_debugging

# Create Inference Map
python script_create_map_template.py \
  --a="$base_dir/ljs_ipa_ms_from_scratch/filelist/symbols.json" \
  --b="examples/ipa/corpora.txt" \
  --mode="infer"
  --ignore_tones
  --ignore_arcs
  --no_debugging

# Update Inference Map
python -m script_create_map_template \
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
