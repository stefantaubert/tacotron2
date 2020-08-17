########################################################################################
# Preprocessing
########################################################################################

# Init

## Capslock Dev
source /datasets/code/tacotron2/configs/envs/dev-caps.sh

## Capslock GCP
source /datasets/code/tacotron2/configs/envs/prod-caps.sh

## GCP
# For usage with a t4 on Google Cloud Plattform
source /home/stefan_taubert/tacotron2/configs/envs/prod-gcp.sh

## Phil
source /home/stefan/tacotron2/configs/envs/prod-phil.sh

# Preprocessing

python -m src.cli.runner preprocess-ljs \
  --path="$ljs_data" \
  --auto_dl \
  --base_dir="$base_dir" \
  --ds_name="ljs"
