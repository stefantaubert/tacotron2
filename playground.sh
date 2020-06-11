base_dir='/datasets/models/taco2pt_ms'
checkpoint_name='2999'
checkpoint='thchs_C17_ipa'
model_path=${base_dir}/output/checkpoint_${checkpoint_name}
dest_model_path=${base_dir}/saved_checkpoints/${checkpoint}_${checkpoint_name}
echo $model_path
echo $dest_model_path