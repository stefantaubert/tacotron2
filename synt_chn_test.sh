base_dir='/datasets/models/taco2pt_ms'
checkpoint='thchs_A11_ipa_2500'
speaker='A11'

python paths.py --base_dir=$base_dir

# IPA
#python script_txt_pre.py --base_dir=$base_dir --ipa='true' --text='examples/north_sven_orig.txt' --is_ipa='true'
#python synthesize.py --base_dir=$base_dir --checkpoint=$checkpoint --output_name='north_sven_orig'

# python script_txt_pre.py --base_dir=$base_dir --ipa='true' --text='examples/north_sven_v1.txt' --is_ipa='true'
# python synthesize.py --base_dir=$base_dir --checkpoint=$checkpoint --output_name='north_sven_v1'

python script_txt_pre.py --base_dir=$base_dir --ipa='true' --text='examples/north_chn.txt' --is_ipa='true' --speaker=$speaker
python synthesize.py --base_dir=$base_dir --checkpoint=$checkpoint --output_name='north_chn' --speaker=$speaker
