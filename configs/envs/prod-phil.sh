# Init
# usage: source path/to/this/file
## Capslock Dev
cd /home/stefan/tacotron2
#source activate taco2pytorch
conda activate test2
export base_dir="/home/stefan/taco2pt_v2"
export datasets_dir="/home/stefan/datasets"
export pretrained_dir="$base_dir/pretrained"
export waveglow="$pretrained_dir/waveglow_256channels_universal_v5.pt"
export ljs_data="$datasets_dir/LJSpeech-1.1"
export thchs_original_data="$datasets_dir/thchs"
export thchs_data="$datasets_dir/thchs_16bit_22050kHz"