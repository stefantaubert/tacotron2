# usage: source path/to/this/file
cd /datasets/code/tacotron2-dev
conda activate test1

datasets_dir="/datasets"
export base_dir="$datasets_dir/gcp_home"
export pretrained_dir="$datasets_dir/models/pretrained"
export waveglow="$pretrained_dir/waveglow_256channels_universal_v5.pt"

export ljs_data="$datasets_dir/LJSpeech-1.1"
export thchs_original_data="$datasets_dir/thchs"
export thchs_data="$datasets_dir/thchs_16bit_22050kHz"