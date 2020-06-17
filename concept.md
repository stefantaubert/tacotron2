# Filestructure

$base_dir/ds
- ljs_ipa/
  - 1/
- ljs_en/
- thchs_ipa/
  - A11/
    - symbols.log
    - symbols.json
    - filelist_log.csv
    - filelist.csv
- thchas_toneless_ipa/

$base_dir/training_2020-06-17_11-11-03
- logs/
- inference/
  - north_sven_v2_2020-06-17_11-11-03/
    - config.log
    - input.log
    - input_map.log
    - 1_input_normalized_sentences.txt
    - 2_input_sentences.txt
    - 3_input_sentences_mapped.txt
    - 4_input_symbols.txt
    - output.wav
  - ...
- analysis/
  - sims.log
  - 3d.html
  - 2d.html
- filelist/
  - weights.npy
  - symbols.json
  - symbols.txt
  - audio_text_train_filelist.csv
  - audio_text_test_filelist.csv
  - audio_text_val_filelist.csv
- checkpoints/
- train_config.log
- description.txt

# Configs
train.json
{
  "seed": 1234,
  "hparams"
  "warm_start"
  "checkpoint_path"
  "use_pretrained_weights"
  "pretrained_model"
  "pretrained_model_symbols"
  "ds_name"
  "speaker"
  "map_mode": separate,unify,map
  "map"
  %"n_gpus"
  %"rank"
  %"group_name"
}

north_sven_v2.json
{
  "ipa"
  "text"
  "is_ipa"
  "map"
  "subset_id"
  "hparams"
  "waveglow"
  "speaker"
  "ds_name"
  "checkpoint"
}