
# Preprocessing

```bash
export base_dir="/home/stefan_taubert/taco2pt_v2"
export custom_training_name="ljs_ipa_from_scratch"
python paths.py --debug='false' --base_dir=$base_dir --custom_training_name=$custom_training_name
```

# Create ljs_ipa

```bash
export ljs_dir="/home/stefan_taubert/datasets/ljs"
export ds_name="ljs_ipa"
python script_ljs_pre.py --debug='false' --base_dir=$base_dir --data_dir=$ljs_dir --ipa='true' --ds_name=$ds_name --ignore_arcs='true'
```

# Start training

```bash
python script_train.py --debug=false --base_dir=$base_dir --training_dir=$custom_training_name --continue_training=false --warm_start=false --ds_name=ljs_ipa --speaker=1 --hparams=batch_size=26,iters_per_checkpoint=500
```
