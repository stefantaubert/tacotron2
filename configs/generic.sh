# Validate
python script_dl_waveglow_pretrained.py \
  --pretrained_dir=$pretrained_dir \
  --no_debugging

export utterance="random-val"
python script_validate.py --no_debugging --base_dir=$base_dir --training_dir=$custom_training_name --waveglow=$waveglow --utterance=$utterance --custom_checkpoint=''

# Inference IPA
export text="examples/ipa/north_sven_orig.txt"
python -m script_inference --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --lang=ipa --ignore_tones --ignore_arcs --speaker=$speaker --waveglow=$waveglow --map=$text_map --no_debugging --custom_checkpoint=''

export text="examples/chn/thchs.txt"
python -m script_inference --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --lang=chn --ignore_tones --ignore_arcs --speaker=$speaker --waveglow=$waveglow --map=$text_map --no_debugging --custom_checkpoint=''

export text="examples/ger/example.txt"
python -m script_inference --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --lang=ger --ignore_tones --ignore_arcs --speaker=$speaker --waveglow=$waveglow --map=$text_map --no_debugging --custom_checkpoint=''

export text="examples/ger/nord.txt"
python -m script_inference --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --lang=ger --ignore_tones --ignore_arcs --speaker=$speaker --waveglow=$waveglow --map=$text_map --no_debugging --custom_checkpoint=''

export text="examples/ipa/north_sven_v2.txt"
python -m script_inference --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --lang=ipa --ignore_tones --ignore_arcs --speaker=$speaker --waveglow=$waveglow --map=$text_map --no_debugging --custom_checkpoint=''

export text="examples/en/north.txt"
python -m script_inference --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --lang=en --ignore_tones --ignore_arcs --speaker=$speaker --waveglow=$waveglow --map=$text_map --no_debugging --custom_checkpoint=''

export text="examples/en/democritus_v2.txt"
python -m script_inference --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --lang=en --ignore_tones --ignore_arcs --speaker=$speaker --waveglow=$waveglow --map=$text_map --no_debugging --custom_checkpoint=''

export text="examples/ipa/north_ger.txt"
python -m script_inference --base_dir=$base_dir --training_dir=$custom_training_name --ipa --text=$text --lang=ipa --ignore_tones --ignore_arcs --speaker=$speaker --waveglow=$waveglow --map=$text_map --no_debugging --custom_checkpoint=''

# Validate checkpoints
export hparams="batch_size=$batch_size"
export select_pattern=10000
python -m script_eval_checkpoints --base_dir=$base_dir --training_dir=$custom_training_name --hparams=$hparams --select=2000 --min=70000 --no_debugging

# Plot Embeddings
python -m script_plot_embeddings \
  --base_dir=$base_dir \
  --training_dir=$custom_training_name \
  --no_debugging \
  --custom_checkpoint=''

# Create Inference Map
python script_create_map_template.py \
  --a="$base_dir/$custom_training_name/filelist/symbols.json" \
  --b="examples/ipa/corpora.txt" \
  --mode="infer"
  --ignore_tones
  --ignore_arcs
  --no_debugging

# Update Inference Map
python -m script_create_map_template \
  --a="$base_dir/$custom_training_name/filelist/symbols.json" \
  --b="examples/ipa/corpora.txt" \
  --existing_map="maps/inference/en_v1.json"
  --mode="infer"
  --ignore_tones
  --ignore_arcs
  --no_debugging
