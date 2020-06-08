base_dir='/datasets/models/taco2pt_ipa'
checkpoint='checkpoint_49000'

python paths.py --base_dir=$base_dir

# IPA
python script_txt_pre.py --base_dir=$base_dir --ipa='true' --text='examples/test_ipa.txt' --is_ipa='true'
python synthesize.py --base_dir=$base_dir --checkpoint=$checkpoint --output_name='test_ipa'
