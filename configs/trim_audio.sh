source /datasets/code/tacotron2/configs/envs/dev-caps.sh
python ./src/pre/audio/remove_silence_script.py \
  --no_debugging \
  --base_dir="$base_dir/analysis/trimming" \
  --wav="$thchs_data/wav/train/A13/A13_228.wav" \
  --chunk_size=5 \
  --threshold_start=-25 \
  --buffer_start_ms=100 \
  --threshold_end=-25 \
  --buffer_end_ms=150

python ./src/pre/audio/remove_silence_script.py \
  --no_debugging \
  --base_dir="$base_dir/analysis/trimming" \
  --wav="$thchs_data/wav/train/C18/C18_725.wav" \
  --chunk_size=5 \
  --threshold_start=-25 \
  --buffer_start_ms=100 \
  --threshold_end=-35 \
  --buffer_end_ms=150

python ./src/pre/audio/remove_silence_script.py \
  --no_debugging \
  --base_dir="$base_dir/analysis/trimming" \
  --wav="$thchs_data/wav/train/B12/B12_482.wav" \
  --chunk_size=5 \
  --threshold_start=-25 \
  --buffer_start_ms=100 \
  --threshold_end=-35 \
  --buffer_end_ms=150
