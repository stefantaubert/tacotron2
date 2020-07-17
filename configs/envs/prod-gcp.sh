# usage: source path/to/this/file
cd /home/stefan_taubert/tacotron2
#source activate taco2pytorch
conda activate taco2pytorch

export base_dir="/home/stefan_taubert/taco2pt_v2"
export pretrained_dir="$base_dir/pretrained"
export waveglow="$pretrained_dir/waveglow_256channels_universal_v5.pt"

datasets_dir="/home/stefan_taubert/datasets"
export ljs_data="$datasets_dir/LJSpeech-1.1"
export thchs_original_data="$datasets_dir/thchs"
export thchs_data="$datasets_dir/thchs_16bit_22050kHz"