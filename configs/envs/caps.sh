# usage: source path/to/this/file
code_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../..
cd $code_dir
export PYTHONPATH=$code_dir

datasets_dir="/datasets"
# required
export base_dir="$datasets_dir/models/taco2pt_v5"

# optional
export ljs_data="$datasets_dir/LJSpeech-1.1"

# optional
export thchs_data="$datasets_dir/thchs_wav"

# optional
export arctic_data="$datasets_dir/l2arctic"

# optional
export lbritts_data="$datasets_dir/libriTTS"

# optional
export nnlv_pilot_data="$datasets_dir/NNLV_pilot"

pipenv shell