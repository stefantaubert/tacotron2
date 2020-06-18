base_dir='/datasets/models/taco2pt_ms'
speaker='1'
ds_name='ljs_ipa'

python script_split_ds.py --base_dir=$base_dir --ds_name=$ds_name --seed='1234' --speaker=$speaker
tensorboard --logdir='${base_dir}/logs'
python train.py --base_dir=$base_dir --checkpoint_path='/datasets/models/pretrained/tacotron2_statedict.pt' --warm_start='true' --n_gpus=1 --rank=0 --group_name='group_name' --hparams='batch_size=26,iters_per_checkpoint=500,epochs=500' --ds_name=$ds_name --speaker=$speaker
