base_dir='/datasets/models/taco2pt_ms'
ds_name='ljs_en'

python paths.py --base_dir=$base_dir
python script_ljs_pre.py --base_dir=$base_dir --data_dir='/datasets/LJSpeech-1.1' --ds_name=$ds_name --ipa='false'