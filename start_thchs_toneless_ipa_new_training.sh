base_dir='/datasets/models/taco2pt_v2'
custom_name='training_2020-06-19_13-47-04'

training_dir=$(python paths.py --base_dir=$base_dir --debug='false' --custom_training_name=$custom_name)

#python run.py --base_dir=$base_dir --training_dir=$training_dir --debug='false' --mode='train' --config='configs/thchs_toneless_ipa/train.json'

python run.py --base_dir=$base_dir --training_dir=$training_dir --debug='false' --mode='train' --config='configs/thchs_toneless_ipa/train_unify.json'

