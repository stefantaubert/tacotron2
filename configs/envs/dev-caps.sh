# usage: source /datasets/code/tacotron2/dev-caps.sh
code_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../..
cd $code_dir
export PYTHONPATH=$code_dir
conda activate py37

datasets_dir="/datasets"
export base_dir="$datasets_dir/models/taco2pt_v3"
export waveglow="$datasets_dir/models/pretrained/waveglow_256channels_universal_v5.pt"

export ljs_data="$datasets_dir/LJSpeech-1.1"
export thchs_data="$datasets_dir/thchs_wav"