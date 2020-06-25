base_dir='/datasets/models/taco2pt_v2'
data_dir=''
ds_name='ljs_ipa'

python paths.py --debug='false' --base_dir=$base_dir --custom_training_name="ljs_ipa_from_scratch"
python script_ljs_pre.py --debug='false' --base_dir=$base_dir --data_dir='/datasets/LJSpeech-1.1' --ds_name=$ds_name --ipa='true'