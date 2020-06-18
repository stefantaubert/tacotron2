base_dir='/datasets/models/taco2pt_ipa'
checkpoint='checkpoint_1000'

# IPA
# unknown: ɾ
python script_txt_pre.py --base_dir=$base_dir --ipa='true' --text='examples/north_narrow.txt' --is_ipa='true'
python script_txt_pre.py --base_dir=$base_dir --ipa='true' --text='examples/north_broad.txt' --is_ipa='true'
# English
python script_txt_pre.py --base_dir=$base_dir --ipa='true' --text='examples/north.txt' --is_ipa='false'
python script_txt_pre.py --base_dir=$base_dir --ipa='true' --text='examples/stella.txt' --is_ipa='false'
python script_txt_pre.py --base_dir=$base_dir --ipa='true' --text='examples/democritus_acc.txt' --is_ipa='false'
python script_txt_pre.py --base_dir=$base_dir --ipa='true' --text='examples/democritus.txt' --is_ipa='false'
python script_txt_pre.py --base_dir=$base_dir --ipa='true' --text='examples/coma.txt' --is_ipa='false'
python script_txt_pre.py --base_dir=$base_dir --ipa='true' --text='examples/rainbow.txt' --is_ipa='false'
python script_txt_pre.py --base_dir=$base_dir --ipa='true' --text='examples/grandfather.txt' --is_ipa='false'
