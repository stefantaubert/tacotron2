base_dir='/datasets/models/taco2pt_ms'
waveglow='/datasets/models/pretrained/waveglow_256channels_universal_v5.pt'
ds_name='thchs'
speaker='C17'
checkpoint_name='2999'
checkpoint=${ds_name}_${speaker}_ipa_${checkpoint_name}

python paths.py --base_dir=$base_dir
model_path=${base_dir}/output/checkpoint_${checkpoint_name}
dest_model_path=${base_dir}/saved_checkpoints/${checkpoint}

if [ ! -f $dest_model_path ]; then
  cp $model_path $dest_model_path
fi

python script_txt_pre.py --base_dir=$base_dir --ipa='true' --text='examples/chn_ipa.txt' --is_ipa='true' --speaker=$speaker --ds_name=$ds_name --speaker=$speaker --debug='false'

python synthesize.py --base_dir=$base_dir --checkpoint=$checkpoint --output_name=chn_${ds_name}_${speaker}_${checkpoint_name} --waveglow=$waveglow --ds_name=$ds_name --speaker=$speaker --hparams='sampling_rate=18000' --debug='false'

python script_txt_pre.py --base_dir=$base_dir --ipa='true' --text='examples/north_chn.txt' --is_ipa='true' --speaker=$speaker --ds_name=$ds_name --speaker=$speaker --map='maps/en_chn.txt' --debug='false'

python synthesize.py --base_dir=$base_dir --checkpoint=$checkpoint --output_name=north_chn_${ds_name}_${speaker}_${checkpoint_name} --waveglow=$waveglow --ds_name=$ds_name --speaker=$speaker --hparams='' --debug='false'
