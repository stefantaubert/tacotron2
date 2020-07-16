# Init
# usage: source path/to/this/file
## Capslock Dev
cd /datasets/code/tacotron2-dev
conda activate test1
export datasets_dir="/datasets"
export base_dir="$datasets_dir/models/taco2pt_v2"
export pretrained_dir="$datasets_dir/models/pretrained"
export waveglow="$pretrained_dir/waveglow_256channels_universal_v5.pt"
