base_dir='/datasets/models/taco2pt_ipa'
ds_dir='/datasets/LJSpeech-1.1'
ipa=True

python paths.py --base_dir=$base_dir
python script_ds_pre.py --base_dir=$base_dir --ljspeech=$ds_dir --ipa=$ipa
python script_split_ds.py --base_dir=$base_dir
python script_txt_pre.py --base_dir=$base_dir --ipa=$ipa