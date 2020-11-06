# usage: source path/to/this/file
# For usage with a t4 on Google Cloud Plattform
code_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../..
cd $code_dir
export PYTHONPATH=$code_dir
conda activate taco2pytorch

# required
export base_dir="/home/stefan_taubert/taco2pt_v5"

datasets_dir="/home/stefan_taubert/datasets"

# optional
export ljs_data="$datasets_dir/LJSpeech-1.1"

# optional
export thchs_data="$datasets_dir/thchs"

# optional
export arctic_data="$datasets_dir/arctic"

# optional
export lbritts_data="$datasets_dir/libriTTS"

# optional
export nnlv_pilot_data="$datasets_dir/NNLV_pilot"