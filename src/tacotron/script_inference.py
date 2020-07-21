import argparse
import json
import os
from shutil import copyfile

from src.common.train_log import reset_log
from src.common.utils import parse_ds_speaker
from src.script_paths import (ds_preprocessed_file_name,
                              ds_preprocessed_symbols_name, filelist_file_name,
                              filelist_symbols_file_name, get_ds_dir,
                              get_filelist_dir, get_inference_dir,
                              inference_config_file, log_inference_config,
                              log_input_file, log_map_file, log_train_config,
                              log_train_map, train_config_file)
from src.tacotron.prepare_ds import prepare
from src.tacotron.script_plot_embeddings import analyse
from src.tacotron.synthesize import infer
from src.tacotron.train import get_last_checkpoint, start_train
from src.tacotron.txt_pre import process_input_text

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--no_debugging', action='store_true')
  parser.add_argument('--base_dir', type=str, help='base directory')
  parser.add_argument('--training_dir', type=str)
  parser.add_argument('--ipa', action='store_true')
  parser.add_argument('--text', type=str)
  parser.add_argument('--lang', type=str, choices=["ipa", "en", "chn", "ger"])
  parser.add_argument('--ignore_tones', action='store_true')
  parser.add_argument('--ignore_arcs', action='store_true')
  parser.add_argument('--map', type=str)
  parser.add_argument('--speaker', type=str)
  parser.add_argument('--subset_id', type=str)
  parser.add_argument('--hparams', type=str)
  parser.add_argument('--waveglow', type=str)
  parser.add_argument('--custom_checkpoint', type=str)
  parser.add_argument('--analysis', action='store_true')

  args = parser.parse_args()

  if not args.no_debugging:
    args.base_dir = '/datasets/gcp_home'
    args.training_dir = 'ljs_ipa_ms_from_scratch'
    args.ipa = True
    # args.text = "examples/chn/thchs.txt"
    # args.lang = "chn"
    # args.text = "examples/ger/nord.txt"
    # args.lang = "ger"
    args.text = "examples/ipa/north_sven_orig.txt"
    args.lang = "ipa"
    #args.map = "maps/inference/chn_v1.json"
    args.map = "maps/inference/en_v1.json"
    args.ignore_tones = True
    args.ignore_arcs = True
    #args.speakers = 'thchs_v5,B2;thchs_v5,A2'
    #args.speaker = 'ljs_ipa_v2,1'
    args.speaker = 'ljs_ipa_v2,1'
    #args.waveglow = "/datasets/models/pretrained/waveglow_256channels_universal_v5.pt"
    args.waveglow = "/datasets/models/pretrained/waveglow_256channels_universal_v5_out.pt"

  training_dir_path = os.path.join(args.base_dir, args.training_dir)

  assert os.path.isfile(args.text)
  assert os.path.isfile(args.waveglow)

  print("Infering text from:", args.text)
  input_name = os.path.splitext(os.path.basename(args.text))[0]
  if args.custom_checkpoint:
    checkpoint = args.custom_checkpoint
  else:
    checkpoint = get_last_checkpoint(training_dir_path)
  speaker = parse_ds_speaker(args.speaker)[1]
  infer_dir_path = get_inference_dir(training_dir_path, input_name, checkpoint, speaker)
  log_inference_config(infer_dir_path, args)
  log_input_file(infer_dir_path, args.text)

  if args.map:
    assert os.path.isfile(args.map)
    print("Using mapping from:", args.map)
    log_map_file(infer_dir_path, args.map)
  else:
    print("Using no mapping.")

  process_input_text(training_dir_path, infer_dir_path, ipa=args.ipa, ignore_tones=args.ignore_tones, ignore_arcs=args.ignore_arcs, subset_id=args.subset_id, lang=args.lang, use_map=bool(args.map))
  infer(training_dir_path, infer_dir_path, hparams=args.hparams, waveglow=args.waveglow, checkpoint=checkpoint, speaker=args.speaker, analysis=args.analysis)
