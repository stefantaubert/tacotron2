import argparse
import json
import os
from shutil import copyfile

from src.common.train_log import reset_log
from src.common.utils import parse_ds_speaker
from src.paths import (ds_preprocessed_file_name,
                              ds_preprocessed_symbols_name, filelist_file_name,
                              filelist_symbols_file_name, get_ds_dir,
                              get_filelist_dir, get_inference_dir, get_checkpoint_dir,
                              inference_config_file, log_inference_config,
                              log_input_file, log_map_file, log_train_config,
                              log_train_map, train_config_file)
from src.tacotron.prepare_ds import prepare
from src.tacotron.plot_embeddings import analyse
from src.synthesize import infer
from src.tacotron.train import start_train
from src.tacotron.txt_pre import process_input_text
from src.common.utils import get_last_checkpoint


def init_inference_parser(parser):
  parser.add_argument('--base_dir', type=str, help='base directory', required=True)
  parser.add_argument('--training_dir', type=str, required=True)
  parser.add_argument('--ipa', action='store_true')
  parser.add_argument('--text', type=str, required=True)
  parser.add_argument('--lang', type=str, choices=["ipa", "en", "chn", "ger"], required=True)
  parser.add_argument('--ignore_tones', action='store_true')
  parser.add_argument('--ignore_arcs', action='store_true')
  parser.add_argument('--weights_map', type=str)
  parser.add_argument('--speaker', type=str, required=True)
  #parser.add_argument('--subset_id', type=str)
  parser.add_argument('--hparams', type=str)
  parser.add_argument('--waveglow', type=str, required=True)
  parser.add_argument('--custom_checkpoint', type=str)
  parser.add_argument('--sentence_pause_s', type=float, default=0.5)
  parser.add_argument('--sigma', type=float, default=0.666)
  parser.add_argument('--denoiser_strength', type=float, default=0.01)
  parser.add_argument('--sampling_rate', type=float, default=22050)
  parser.add_argument('--analysis', action='store_true')
  return __main


def __main(base_dir: str, training_dir: str, ipa: bool, text: str, lang: str, ignore_tones: bool, ignore_arcs:bool, weights_map: str, speaker: str, hparams: str, waveglow: str, custom_checkpoint: str, sentence_pause_s: float, sigma: float, denoiser_strength: float, sampling_rate: float, analysis: bool):
  training_dir_path = os.path.join(base_dir, training_dir)

  assert os.path.isfile(text)
  assert os.path.isfile(waveglow)

  print("Infering text from:", text)
  input_name = os.path.splitext(os.path.basename(text))[0]
  checkpoint_dir = get_checkpoint_dir(training_dir_path)
  if custom_checkpoint:
    checkpoint = custom_checkpoint
  else:
    checkpoint = get_last_checkpoint(checkpoint_dir)
  checkpoint_path = os.path.join(get_checkpoint_dir(training_dir_path), checkpoint)
  
  speaker = parse_ds_speaker(speaker)[1]
  infer_dir_path = get_inference_dir(training_dir_path, input_name, checkpoint, speaker)
  # TODO logging
  #log_inference_config(infer_dir_path, args)
  log_input_file(infer_dir_path, text)

  if weights_map:
    assert os.path.isfile(weights_map)
    print("Using mapping from:", weights_map)
    log_map_file(infer_dir_path, weights_map)
  else:
    print("Using no mapping.")

  process_input_text(
    training_dir_path,
    infer_dir_path,
    ipa=ipa,
    ignore_tones=ignore_tones,
    ignore_arcs=ignore_arcs,
    subset_id=0,
    lang=lang,
    use_map=bool(weights_map)
  )

  infer(
    training_dir_path=training_dir_path,
    infer_dir_path=infer_dir_path,
    hparams=hparams,
    waveglow=waveglow,
    checkpoint_path=checkpoint_path,
    speaker=speaker,
    analysis=analysis,
    sentence_pause_s=sentence_pause_s,
    sigma=sigma,
    denoiser_strength=denoiser_strength,
    sampling_rate=sampling_rate
  )


if __name__ == "__main__":
  __main(
    base_dir = '/datasets/models/taco2pt_v2',
    #training_dir = 'ljs_ipa_ms_from_scratch',
    training_dir = 'thchs_ipa_warm_mapped_all_tones',
    ipa = True,
    text = "examples/chn/north.txt",
    lang = "chn",
    #text = "examples/ger/nord.txt",
    #lang = "ger",
    #text = "examples/ipa/north_sven_orig.txt",
    #lang = "ipa",
    #text = "examples/en/ljs_0001.txt",
    #lang = "en",
    map = "maps/inference/chn_v1.json",
    #map = "maps/inference/en_v1.json",
    ignore_tones = False,
    ignore_arcs = True,
    #speakers = 'thchs_v5,B2;thchs_v5,A2',
    #speaker = 'ljs_ipa_v2,1',
    speaker = 'thchs_nosil_tones,D31',
    #waveglow = "/datasets/models/pretrained/waveglow_256channels_universal_v5.pt",
    waveglow = "/datasets/models/pretrained/waveglow_256channels_universal_v5_out.pt",
    analysis = True,
    #denoiser_strength = 0.5,
    sigma = 0.666,
  )