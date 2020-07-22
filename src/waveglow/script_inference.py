import argparse
import os
from shutil import copyfile

from src.common.train_log import reset_log
from src.script_paths import get_inference_dir
from src.waveglow.prepare_ds import load_filepaths
from src.waveglow.train import get_last_checkpoint
from src.waveglow.inference import infer

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--no_debugging', action='store_true')
  parser.add_argument('--base_dir', type=str, help='base directory')
  parser.add_argument('--training_dir', type=str)
  parser.add_argument('--wav', type=str)
  parser.add_argument('--hparams', type=str)
  parser.add_argument("--denoiser_strength", default=0.0, type=float, help='Removes model bias. Start with 0.1 and adjust')
  parser.add_argument("--sigma", default=1.0, type=float)
  parser.add_argument('--custom_checkpoint', type=str)

  args = parser.parse_args()

  if not args.no_debugging:
    args.base_dir = '/datasets/models/taco2pt_v2'
    args.training_dir = 'wg_debug'
    args.wav = "/datasets/LJSpeech-1.1-test/wavs/LJ001-0100.wav"

  training_dir_path = os.path.join(args.base_dir, args.training_dir)

  checkpoint = args.custom_checkpoint if args.custom_checkpoint else get_last_checkpoint(training_dir_path)

  wav_name = os.path.basename(args.wav)[:-4]

  infer(
    training_dir_path=training_dir_path,
    infer_dir_path=get_inference_dir(training_dir_path, wav_name, checkpoint, ''),
    hparams=args.hparams,
    checkpoint=checkpoint,
    infer_wav_path=args.wav,
    denoiser_strength=args.denoiser_strength,
    sigma=args.sigma
  )
