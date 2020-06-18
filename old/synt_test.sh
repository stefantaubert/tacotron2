base_dir='/datasets/models/taco2pt_testing'
checkpoint='checkpoint_15500'

python paths.py --base_dir=$base_dir

# IPA
# python script_txt_pre.py --base_dir=$base_dir --ipa='true' --text='examples/test_ipa_pinyin.txt' --is_ipa='true'
python script_txt_pre.py --base_dir=$base_dir --ipa='true' --text='examples/north_narrow.txt' --is_ipa='true'
python synthesize.py --base_dir=$base_dir --checkpoint=$checkpoint --output_name='test_ipa'
