
# Validate
python ./src/waveglow/script_dl_pretrained.py \
  --destination=$waveglow \
  --auto_convert \
  --no_debugging

export utterance="random-val"
python ./src/tacotron/script_validate.py --no_debugging --base_dir=$base_dir --training_dir=$custom_training_name --waveglow=$waveglow --utterance=$utterance --custom_checkpoint=''

# Inference IPA
export text="examples/ipa/north_sven_orig.txt"
python ./src/tacotron/script_inference.py --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --lang=ipa --ignore_tones --ignore_arcs --speaker=$speaker --waveglow=$waveglow --map=$text_map --no_debugging --analysis --custom_checkpoint=''

export text="examples/ipa/north_ger.txt"
python ./src/tacotron/script_inference.py --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --lang=ipa --ignore_tones --ignore_arcs --speaker=$speaker --waveglow=$waveglow --map=$text_map --no_debugging --analysis --custom_checkpoint=''

export text="examples/ipa/north_sven_v2.txt"
python ./src/tacotron/script_inference.py --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --lang=ipa --ignore_tones --ignore_arcs --speaker=$speaker --waveglow=$waveglow --map=$text_map --no_debugging --analysis --custom_checkpoint=''

# Inference CHN

export text="examples/chn/thchs.txt"
python ./src/tacotron/script_inference.py --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --lang=chn --ignore_tones --ignore_arcs --speaker=$speaker --waveglow=$waveglow --map=$text_map --no_debugging --analysis --custom_checkpoint=''

# Inference GER

export text="examples/ger/example.txt"
python ./src/tacotron/script_inference.py --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --lang=ger --ignore_tones --ignore_arcs --speaker=$speaker --waveglow=$waveglow --map=$text_map --no_debugging --analysis --custom_checkpoint=''

export text="examples/ger/nord.txt"
python ./src/tacotron/script_inference.py --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --lang=ger --ignore_tones --ignore_arcs --speaker=$speaker --waveglow=$waveglow --map=$text_map --no_debugging --analysis --custom_checkpoint=''

# Inference EN

export text="examples/en/north.txt"
python ./src/tacotron/script_inference.py --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --lang=en --ignore_tones --ignore_arcs --speaker=$speaker --waveglow=$waveglow --map=$text_map --no_debugging --analysis --custom_checkpoint=''

export text="examples/en/democritus_v2.txt"
python ./src/tacotron/script_inference.py --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --lang=en --ignore_tones --ignore_arcs --speaker=$speaker --waveglow=$waveglow --map=$text_map --no_debugging --analysis --custom_checkpoint=''

# Validate checkpoints
export hparams="batch_size=$batch_size"
export select_pattern=10000
python ./src/tacotron/script_eval_checkpoints.py --base_dir=$base_dir --training_dir=$custom_training_name --hparams=$hparams --no_debugging --select=2000 --min=70000

# Plot Embeddings
python ./src/tacotron/script_plot_embeddings.py \
  --base_dir=$base_dir \
  --training_dir=$custom_training_name \
  --no_debugging \
  --custom_checkpoint=''

# Create Inference Map
python ./src/tacotron/script_create_map_template.py \
  --a="$base_dir/$custom_training_name/filelist/symbols.json" \
  --b="examples/ipa/corpora.txt" \
  --mode="infer"
  --ignore_tones
  --ignore_arcs
  --no_debugging

# Update Inference Map
python ./src/tacotron/script_create_map_template.py \
  --a="$base_dir/$custom_training_name/filelist/symbols.json" \
  --b="examples/ipa/corpora.txt" \
  --existing_map="maps/inference/en_v1.json"
  --mode="infer"
  --ignore_tones
  --ignore_arcs
  --no_debugging

## Open Tensorboard
tensorboard --logdir=$base_dir/$custom_training_name/logs