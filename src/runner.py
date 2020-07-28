import argparse

if __name__ == "__main__":
  main_parser = argparse.ArgumentParser()
  subparsers = main_parser.add_subparsers(help='sub-command help')

  from src.paths import main as handler
  parser = subparsers.add_parser('paths', help='preprocess help')
  parser.add_argument('--base_dir', type=str, help='base directory')
  parser.add_argument('--custom_training_name', type=str)
  parser.set_defaults(invoke_handler=handler)

  from src.pre.ljs.pre import main as handler
  parser = subparsers.add_parser('ljs-pre')
  parser.add_argument('--base_dir', type=str, help='base directory')
  parser.add_argument('--data_dir', type=str, help='LJSpeech dataset directory')
  parser.add_argument('--ipa', action='store_true', help='transcribe to IPA')
  parser.add_argument('--ignore_arcs', action='store_true')
  parser.add_argument('--ds_name', type=str, help='the name you want to call the dataset')
  parser.add_argument('--auto_dl', action='store_true')
  parser.set_defaults(invoke_handler=handler)

  from src.pre.thchs.dl import main as handler
  parser = subparsers.add_parser('thchs-dl')
  parser.add_argument('--kaldi_version', action='store_true')
  parser.add_argument('--data_dir', type=str, help='THCHS dataset directory')
  parser.set_defaults(invoke_handler=handler)
  
  from src.pre.thchs.pre import preprocess as handler
  parser = subparsers.add_parser('thchs-pre')
  parser.add_argument('--base_dir', type=str, help='base directory')
  parser.add_argument('--kaldi_version', action='store_true')
  parser.add_argument('--data_dir', type=str, help='THCHS dataset directory')
  parser.add_argument('--ignore_tones', action='store_true')
  parser.add_argument('--ignore_arcs', action='store_true')
  parser.add_argument('--ds_name', type=str, help='the name you want to call the dataset')
  parser.set_defaults(invoke_handler=handler)
  
  from src.pre.thchs.remove_silence import main as handler
  parser = subparsers.add_parser('thchs-remove-silence')
  parser.add_argument('--data_src_dir', type=str, help='THCHS dataset directory')
  parser.add_argument('--data_dest_dir', type=str, help='THCHS destination directory')
  parser.add_argument('--kaldi_version', action='store_true')
  parser.add_argument('--chunk_size', type=int)
  parser.add_argument('--threshold_start', type=float)
  parser.add_argument('--threshold_end', type=float)
  parser.add_argument('--buffer_start_ms', type=float, help="amount of factors of chunk_size at the beginning and the end should be reserved")
  parser.add_argument('--buffer_end_ms', type=float, help="amount of factors of chunk_size at the beginning and the end should be reserved")
  parser.set_defaults(invoke_handler=handler)
  
  from src.pre.thchs.upsample import ensure_upsampled as handler
  parser = subparsers.add_parser('thchs-upsample')
  parser.add_argument('--data_src_dir', type=str, help='THCHS dataset directory')
  parser.add_argument('--data_dest_dir', type=str, help='THCHS destination directory')
  parser.add_argument('--kaldi_version', action='store_true')
  parser.add_argument('--new_rate', type=int, default=22050)
  parser.set_defaults(invoke_handler=handler)
  
  from src.tacotron.create_map_template import main as handler
  parser = subparsers.add_parser('create-map')
  parser.add_argument('--a', type=str)
  parser.add_argument('--b', type=str)
  parser.add_argument('--existing_map', type=str, help="if your corpora extended and you want to extend an existing symbolsmap.")
  parser.add_argument('--mode', type=str, choices=["weights", "infer"])
  parser.add_argument('--out', type=str, default='/tmp/map.json')
  parser.add_argument('--reverse', action='store_true')
  parser.add_argument('--ignore_tones', action='store_true')
  parser.add_argument('--ignore_arcs', action='store_true')
  parser.set_defaults(invoke_handler=handler)
  
  from src.tacotron.eval_checkpoints import main as handler
  parser = subparsers.add_parser('eval-checkpoints')
  parser.add_argument('--base_dir', type=str, help='base directory')
  parser.add_argument('--training_dir', type=str)
  parser.add_argument('--speakers', type=str)
  parser.add_argument('--hparams', type=str)
  parser.add_argument('--select', type=int)
  parser.add_argument('--min_it', type=int)
  parser.add_argument('--max_it', type=int)
  parser.set_defaults(invoke_handler=handler)
  
  from src.tacotron.plot_embeddings import main as handler
  parser = subparsers.add_parser('plot-embeddings')
  parser.add_argument('--base_dir', type=str, help='base directory')
  parser.add_argument('--training_dir', type=str)
  parser.add_argument('--custom_checkpoint', type=str)
  parser.set_defaults(invoke_handler=handler)
  
  from src.tacotron.train import main as handler
  parser = subparsers.add_parser('tacotron-train')
  parser.add_argument('--base_dir', type=str, help='base directory')
  parser.add_argument('--training_dir', type=str)
  parser.add_argument('--continue_training', action='store_true')
  parser.add_argument('--seed', type=int, default=1234)
  parser.add_argument('--warm_start', action='store_true')
  parser.add_argument('--pretrained_path', type=str)
  parser.add_argument('--speakers', type=str, help="ds_name,speaker_id;... or ds_name,all;...")
  parser.add_argument('--test_size', type=float, default=0.001)
  parser.add_argument('--validation_size', type=float, default=0.1)
  parser.add_argument('--hparams', type=str)
  parser.add_argument('--pretrained_model', type=str)
  parser.add_argument('--pretrained_model_symbols', type=str)
  parser.add_argument('--weight_map_mode', type=str, choices=['', 'same_symbols_only', 'use_map'])
  parser.add_argument('--inference_map', type=str)
  parser.set_defaults(invoke_handler=handler)
  
  from src.tacotron.validate import main as handler
  parser = subparsers.add_parser('tacotron-validate')
  parser.add_argument('--base_dir', type=str, help='base directory')
  parser.add_argument('--training_dir', type=str)
  parser.add_argument('--utterance', type=str, help="Utterance name or random-val or random-val-B12")
  parser.add_argument('--hparams', type=str)
  parser.add_argument('--waveglow', type=str)
  parser.add_argument('--custom_checkpoint', type=str)
  parser.set_defaults(invoke_handler=handler)

  from src.tacotron.inference import main as handler
  parser = subparsers.add_parser('tacotron-infer')
  parser.add_argument('--base_dir', type=str, help='base directory')
  parser.add_argument('--training_dir', type=str)
  parser.add_argument('--ipa', action='store_true')
  parser.add_argument('--text', type=str)
  parser.add_argument('--lang', type=str, choices=["ipa", "en", "chn", "ger"])
  parser.add_argument('--ignore_tones', action='store_true')
  parser.add_argument('--ignore_arcs', action='store_true')
  parser.add_argument('--weights_map', type=str)
  parser.add_argument('--speaker', type=str)
  parser.add_argument('--subset_id', type=str)
  parser.add_argument('--hparams', type=str)
  parser.add_argument('--waveglow', type=str)
  parser.add_argument('--custom_checkpoint', type=str)
  parser.add_argument('--sentence_pause_s', type=float, default=0.5)
  parser.add_argument('--sigma', type=float, default=0.666)
  parser.add_argument('--denoiser_strength', type=float, default=0.01)
  parser.add_argument('--sampling_rate', type=float, default=22050)
  parser.add_argument('--analysis', action='store_true')
  parser.set_defaults(invoke_handler=handler)

  from src.waveglow.converter.convert import convert as handler
  parser = subparsers.add_parser('waveglow-convert')
  parser.add_argument('--source', type=str)
  parser.add_argument('--destination', type=str)
  parser.set_defaults(invoke_handler=handler)

  from src.waveglow.dl_pretrained import convert as handler
  parser = subparsers.add_parser('waveglow-dl')
  parser.add_argument('--destination', type=str)
  parser.add_argument('--auto_convert', action='store_true')
  parser.set_defaults(invoke_handler=handler)

  from src.waveglow.train import main as handler
  parser = subparsers.add_parser('waveglow-train')
  parser.add_argument('--base_dir', type=str, help='base directory')
  parser.add_argument('--training_dir', type=str)
  parser.add_argument('--continue_training', action='store_true')
  parser.add_argument('--seed', type=str, default=1234)
  #parser.add_argument('--pretrained_path', type=str)
  parser.add_argument('--speakers', type=str, help="ds_name,speaker_id;... or ds_name,all;...")
  parser.add_argument('--train_size', type=float, default=0.9)
  parser.add_argument('--validation_size', type=float, default=1.0)
  parser.add_argument('--hparams', type=str)
  parser.set_defaults(invoke_handler=handler)

  from src.waveglow.validate import main as handler
  parser = subparsers.add_parser('waveglow-validate')
  parser.add_argument('--base_dir', type=str, help='base directory')
  parser.add_argument('--training_dir', type=str)
  parser.add_argument('--utterance', type=str, help="Utterance name or random-val or random-val-B12")
  parser.add_argument('--hparams', type=str)
  parser.add_argument("--denoiser_strength", default=0.0, type=float, help='Removes model bias. Start with 0.1 and adjust')
  parser.add_argument("--sigma", default=1.0, type=float)
  parser.add_argument('--custom_checkpoint', type=str)
  parser.set_defaults(invoke_handler=handler)

  from src.waveglow.inference import main as handler
  parser = subparsers.add_parser('waveglow-infer')
  parser.add_argument('--base_dir', type=str, help='base directory')
  parser.add_argument('--training_dir', type=str)
  parser.add_argument('--wav', type=str)
  parser.add_argument('--hparams', type=str)
  parser.add_argument("--denoiser_strength", default=0.0, type=float, help='Removes model bias.')
  parser.add_argument("--sigma", default=1.0, type=float)
  parser.add_argument('--custom_checkpoint', type=int)
  parser.set_defaults(invoke_handler=handler)


  args = main_parser.parse_args()
  
  params = vars(args)
  invoke_handler = params.pop("invoke_handler")
  invoke_handler(**params)
