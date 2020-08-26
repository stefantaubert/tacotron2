# usage: source path/to/this/file
code_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../..
cd $code_dir
export PYTHONPATH=$code_dir
conda activate py37

datasets_dir="/datasets"
export base_dir="$datasets_dir/models/taco2pt_v4"

export ljs_data="$datasets_dir/LJSpeech-1.1"
export thchs_data="$datasets_dir/thchs_wav"