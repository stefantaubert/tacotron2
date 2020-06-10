base_dir='/datasets/models/taco2pt_ms'
checkpoint='ljs_en_1_en_68500'
speaker='1'
ds_name='ljs_en'


python paths.py --base_dir=$base_dir

python script_txt_pre.py --base_dir=$base_dir --ipa='false' --text='examples/north.txt' --is_ipa='false' --speaker=$speaker --ds_name=$ds_name --speaker=$speaker
python synthesize.py --base_dir=$base_dir --checkpoint=$checkpoint --output_name='north_en' --waveglow='/datasets/models/pretrained/waveglow_256channels_universal_v5.pt' --ds_name=$ds_name --speaker=$speaker --hparams='sampling_rate=22050'
