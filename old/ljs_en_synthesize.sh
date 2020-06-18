base_dir='/datasets/models/taco2pt_ms'
waveglow='/datasets/models/pretrained/waveglow_256channels_universal_v5.pt'
speaker='1'
ds_name='ljs_en'
checkpoint_name='51500'
checkpoint=${ds_name}_${checkpoint_name}

python paths.py --base_dir=$base_dir
model_path=${base_dir}/output/checkpoint_${checkpoint_name}
dest_model_path=${base_dir}/saved_checkpoints/${checkpoint}

if [ ! -f $dest_model_path ]; then
  cp $model_path $dest_model_path
fi


python script_txt_pre.py --base_dir=$base_dir --ipa='false' --text='examples/north.txt' --is_ipa='false' --speaker=$speaker --ds_name=$ds_name --speaker=$speaker --debug='false'

python synthesize.py --base_dir=$base_dir --checkpoint=$checkpoint --output_name=north_chn_${ds_name}_${speaker}_${checkpoint_name} --waveglow=$waveglow --ds_name=$ds_name --speaker=$speaker --hparams='' --debug='false'
