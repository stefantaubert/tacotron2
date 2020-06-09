base_dir='/datasets/models/taco2pt_ipa_en_49000'
checkpoint='checkpoint_49000'

python paths.py --base_dir=$base_dir

# IPA
#python script_txt_pre.py --base_dir=$base_dir --ipa='true' --text='examples/north_sven_orig.txt' --is_ipa='true'
#python synthesize.py --base_dir=$base_dir --checkpoint=$checkpoint --output_name='north_sven_orig'

# python script_txt_pre.py --base_dir=$base_dir --ipa='true' --text='examples/north_sven_v1.txt' --is_ipa='true'
# python synthesize.py --base_dir=$base_dir --checkpoint=$checkpoint --output_name='north_sven_v1'

python script_txt_pre.py --base_dir=$base_dir --ipa='true' --text='examples/north_sven_v2.txt' --is_ipa='true'
python synthesize.py --base_dir=$base_dir --checkpoint=$checkpoint --output_name='north_sven_v2'
