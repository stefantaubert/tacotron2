base_dir='/datasets/models/taco2pt_ms'
pretrained='/datasets/models/pretrained/tacotron2_statedict.pt'
speaker='C17'
ds_name='thchs_no_tone'

python script_split_ds.py --base_dir=$base_dir --ds_name=$ds_name --seed='1234' --speaker=$speaker --debug='false' --speaker_based='true'

python train.py --base_dir=$base_dir --checkpoint_path=$pretrained --warm_start='true' --n_gpus=1 --rank=0 --group_name='group_name' --hparams='sampling_rate=16000,batch_size=35,iters_per_checkpoint=100,epochs=500' --debug='false' --use_pretrained_weights='false'
