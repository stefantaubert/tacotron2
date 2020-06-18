base_dir='/datasets/models/taco2pt_v2'
ds_name='thchs_ipa'

python script_thchs_pre.py --base_dir=$base_dir --data_dir='/datasets/thchs_wav' --ds_name=$ds_name --ignore_tones='false' --debug='false'