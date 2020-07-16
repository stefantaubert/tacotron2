# Init
# usage: source path/to/this/file
## Capslock Dev
cd /home/stefan_taubert/tacotron2
#source activate taco2pytorch
conda activate taco2pytorch
export base_dir="/home/stefan_taubert/taco2pt_v2"
export datasets_dir="/home/stefan_taubert/datasets"
export pretrained_dir="$base_dir/pretrained"
export waveglow="$pretrained_dir/waveglow_256channels_universal_v5.pt"