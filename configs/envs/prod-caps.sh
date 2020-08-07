# usage: source path/to/this/file
code_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../..
cd $code_dir
export PYTHONPATH=$code_dir
conda activate py37

datasets_dir="/datasets"
export base_dir="$datasets_dir/gcp_home"
export waveglow="$datasets_dir/models/pretrained/waveglow_256channels_universal_v5.pt"

export ljs_data="$datasets_dir/LJSpeech-1.1"
export thchs_original_data="$datasets_dir/thchs"
export thchs_upsampled_data="$datasets_dir/thchs_16bit_22050kHz"
export thchs_nosil_data="$datasets_dir/thchs_16bit_22050kHz_nosil"