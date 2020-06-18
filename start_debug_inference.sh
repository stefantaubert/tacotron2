base_dir='/datasets/models/taco2pt_v2'
training_dir='debug'

python run.py --base_dir=$base_dir --training_dir=$training_dir --debug='false' --mode='infer' --config='configs/debug/grandfather.json'

python run.py --base_dir=$base_dir --training_dir=$training_dir --debug='false' --mode='infer' --config='configs/debug/north_sven_v2.json'
