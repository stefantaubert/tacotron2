base_dir='/datasets/models/taco2pt_ms'
checkpoint='thchs_A11_ipa_500'
ds_name='thchs'
speaker='A11'

python paths.py --base_dir=$base_dir

python script_txt_pre.py --base_dir=$base_dir --ipa='true' --text='examples/north_chn.txt' --is_ipa='true' --speaker=$speaker --ds_name=$ds_name --speaker=$speaker
python synthesize.py --base_dir=$base_dir --checkpoint=$checkpoint --output_name='north_chn' --waveglow='/datasets/models/pretrained/waveglow_256channels_universal_v5.pt' --ds_name=$ds_name --speaker=$speaker --hparams='sampling_rate=17000' --map='maps/en_chn.txt'
