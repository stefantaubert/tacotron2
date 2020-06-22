base_dir='/datasets/models/taco2pt_v2'
training_dir='thchs_mapped_weights_pretrained-2020-06-22'

#python run.py --base_dir=$base_dir --training_dir=$training_dir --debug='false' --mode='infer' --config='configs/thchs_toneless_ipa/chn_ipa.json'

#python run.py --base_dir=$base_dir --training_dir=$training_dir --debug='false' --mode='infer' --config='configs/thchs_toneless_ipa/grandfather.json'

python run.py --base_dir=$base_dir --training_dir=$training_dir --debug='false' --mode='infer' --config='configs/thchs_toneless_ipa/democritus.json'

#python run.py --base_dir=$base_dir --training_dir=$training_dir --debug='false' --mode='infer' --config='configs/thchs_toneless_ipa/north_sven_v2.json'
