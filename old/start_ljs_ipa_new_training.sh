base_dir='/datasets/models/taco2pt_v2'

training_dir=$(python paths.py --base_dir=$base_dir --debug='false')

python run.py --base_dir=$base_dir --training_dir=$training_dir --debug='false' --mode='train' --config='configs/ljs_ipa/train.json'

