base_dir='/datasets/models/taco2pt_v2'
training_dir='training_2020-06-18_12-37-40'

python run.py --base_dir=$base_dir --training_dir=$training_dir --debug='false' --mode='infer' --config='configs/thchs_toneless_ipa/grandfather.json'

python run.py --base_dir=$base_dir --training_dir=$training_dir --debug='false' --mode='infer' --config='configs/thchs_toneless_ipa/north_sven_v2.json'
