
## Plot Embeddings
python -m src.cli.runner tacotron-plot-embeddings \
  --base_dir=$base_dir \
  --train_name=$train_name

# Validate
python -m src.cli.runner tacotron-validate \
  --base_dir=$base_dir \
  --train_name=$train_name

# Inference
export text="examples/ipa/north_sven_orig.txt"
python -m src.cli.runner tacotron-infer \
  --base_dir=$base_dir \
  --train_name=$train_name \
  --ipa \
  --ds_speaker=$ds_speaker \
  --symbols_map=$text_map \
  --lang=ENG \
  --text=$text \
  --analysis \
  --custom_checkpoint=0
