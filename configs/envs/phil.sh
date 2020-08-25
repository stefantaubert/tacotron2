# usage: source path/to/this/file
code_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../..
cd $code_dir
export PYTHONPATH=$code_dir
conda activate py37

export base_dir="/home/stefan/taco2pt_v2"

datasets_dir="/home/stefan/datasets"
export ljs_data="$datasets_dir/LJSpeech-1.1"
export thchs_data="$datasets_dir/thchs"