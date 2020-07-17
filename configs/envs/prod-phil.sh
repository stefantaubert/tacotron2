# usage: source path/to/this/file
cd /home/stefan/tacotron2
conda activate test2

export base_dir="/home/stefan/taco2pt_v2"
export pretrained_dir="$base_dir/pretrained"
export waveglow="$pretrained_dir/waveglow_256channels_universal_v5.pt"

datasets_dir="/home/stefan/datasets"
export ljs_data="$datasets_dir/LJSpeech-1.1"
export thchs_original_data="$datasets_dir/THCHS-30"
export thchs_data="$datasets_dir/THCHS-30_16bit-22050kHz"