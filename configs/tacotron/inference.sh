# Inference

## IPA
export text="examples/ipa/north_sven_v2.txt"
export text="examples/ipa/north_ger.txt"
export text="examples/ipa/north_sven_orig.txt"
python -m src.cli.runner tacotron-infer \
  --base_dir=$base_dir \
  --train_name=$train_name \
  --ipa \
  --ds_speaker=$ds_speaker \
  --symbols_map="maps/inference/$prep_name.json" \
  --lang=IPA \
  --text=$text \
  --analysis \
  --custom_checkpoint=0

## Chinese
export text="examples/chn/thchs.txt"
export text="examples/chn/north.txt"
python -m src.cli.runner tacotron-infer \
  --base_dir=$base_dir \
  --train_name=$train_name \
  --ipa \
  --ds_speaker=$ds_speaker \
  --symbols_map="maps/inference/$prep_name.json" \
  --lang=CHN \
  --text=$text \
  --analysis \
  --custom_checkpoint=0

## German
export text="examples/ger/example.txt"
export text="examples/ger/nord.txt"
python -m src.cli.runner tacotron-infer \
  --base_dir=$base_dir \
  --train_name=$train_name \
  --ipa \
  --ds_speaker=$ds_speaker \
  --symbols_map="maps/inference/$prep_name.json" \
  --lang=GER \
  --text=$text \
  --analysis \
  --custom_checkpoint=0

## English
export text="examples/en/coma.txt"
export text="examples/en/rainbow.txt"
export text="examples/en/stella.txt"
export text="examples/en/democritus_v2.txt"
export text="examples/en/north.txt"
python -m src.cli.runner tacotron-infer \
  --base_dir=$base_dir \
  --train_name=$train_name \
  --ipa \
  --ds_speaker=$ds_speaker \
  --symbols_map="maps/inference/$prep_name.json" \
  --lang=ENG \
  --text=$text \
  --analysis \
  --custom_checkpoint=0
