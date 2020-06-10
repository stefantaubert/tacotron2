base_dir='/datasets/models/taco2pt_ms'
speaker='C17'
ds_name='thchs'

python script_split_ds.py --base_dir=$base_dir --ds_name=$ds_name --seed='1234' --speaker=$speaker
python train.py --base_dir=$base_dir --checkpoint_path='/datasets/models/pretrained/tacotron2_statedict.pt' --warm_start='true' --n_gpus=1 --rank=0 --group_name='group_name' --hparams='sampling_rate=16000,batch_size=35,iters_per_checkpoint=500,epochs=500' --ds_name=$ds_name --speaker=$speaker
