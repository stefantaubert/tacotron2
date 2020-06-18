base_dir='/datasets/models/taco2pt_v2'

python paths.py --base_dir=$base_dir --debug='false' --custom_training_name='debug'

python run.py --base_dir=$base_dir --training_dir='debug' --debug='false' --mode='train' --config='configs/debug/train.json'

