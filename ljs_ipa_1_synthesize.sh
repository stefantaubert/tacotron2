base_dir='/datasets/models/taco2pt_ms'
checkpoint='ljs_1_ipa_49000'
speaker='1'
ds_name='ljs_ipa'


python paths.py --base_dir=$base_dir

python script_txt_pre.py --base_dir=$base_dir --ipa='true' --text='examples/north_sven_v2.txt' --is_ipa='true' --speaker=$speaker --ds_name=$ds_name --speaker=$speaker
python synthesize.py --base_dir=$base_dir --checkpoint=$checkpoint --output_name='north_sven_v2' --waveglow='/datasets/models/pretrained/waveglow_256channels_universal_v5.pt' --ds_name=$ds_name --speaker=$speaker --hparams='sampling_rate=22050'
