# usage: source path/to/this/file
code_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../..
cd $code_dir
export PYTHONPATH=$code_dir
#source activate taco2pytorch
conda activate taco2pytorch

export base_dir="/home/stefan_taubert/taco2pt_v2"
export waveglow="$base_dir/pretrained/waveglow_256channels_universal_v5.pt"

datasets_dir="/home/stefan_taubert/datasets"
export ljs_data="$datasets_dir/LJSpeech-1.1"
export thchs_original_data="$datasets_dir/thchs"
export thchs_upsampled_data="$datasets_dir/thchs_16bit_22050kHz"
export thchs_nosil_data="$datasets_dir/thchs_16bit_22050kHz_nosil"