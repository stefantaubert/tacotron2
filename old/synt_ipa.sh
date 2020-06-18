base_dir='/datasets/models/taco2pt_ipa'
checkpoint='checkpoint_49000'

python paths.py --base_dir=$base_dir

# IPA
# unknown: ɾ
python script_txt_pre.py --base_dir=$base_dir --ipa='true' --text='examples/north_narrow.txt' --is_ipa='true'
python synthesize.py --base_dir=$base_dir --checkpoint=$checkpoint --output_name='north_narrow'

python script_txt_pre.py --base_dir=$base_dir --ipa='true' --text='examples/north_broad.txt' --is_ipa='true'
python synthesize.py --base_dir=$base_dir --checkpoint=$checkpoint --output_name='north_broad'

# English

python script_txt_pre.py --base_dir=$base_dir --ipa='true' --text='examples/north.txt' --is_ipa='false'
python synthesize.py --base_dir=$base_dir --checkpoint=$checkpoint --output_name='north'

python script_txt_pre.py --base_dir=$base_dir --ipa='true' --text='examples/stella.txt' --is_ipa='false'
python synthesize.py --base_dir=$base_dir --checkpoint=$checkpoint --output_name='stella'

python script_txt_pre.py --base_dir=$base_dir --ipa='true' --text='examples/democritus_acc.txt' --is_ipa='false'
python synthesize.py --base_dir=$base_dir --checkpoint=$checkpoint --output_name='democritus_acc'

python script_txt_pre.py --base_dir=$base_dir --ipa='true' --text='examples/democritus.txt' --is_ipa='false'
python synthesize.py --base_dir=$base_dir --checkpoint=$checkpoint --output_name='democritus'

python script_txt_pre.py --base_dir=$base_dir --ipa='true' --text='examples/coma.txt' --is_ipa='false'
python synthesize.py --base_dir=$base_dir --checkpoint=$checkpoint --output_name='coma'

python script_txt_pre.py --base_dir=$base_dir --ipa='true' --text='examples/rainbow.txt' --is_ipa='false'
python synthesize.py --base_dir=$base_dir --checkpoint=$checkpoint --output_name='rainbow'

python script_txt_pre.py --base_dir=$base_dir --ipa='true' --text='examples/grandfather.txt' --is_ipa='false'
python synthesize.py --base_dir=$base_dir --checkpoint=$checkpoint --output_name='grandfather'
