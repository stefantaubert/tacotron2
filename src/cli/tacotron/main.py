import os

from tqdm import tqdm

from src.cli import add_console_and_file_out_to_logger, reset_log
from src.cli.pre import (load_filelist, load_filelist_speakers_json,
                         load_filelist_symbol_converter)
from src.cli.tacotron.io import (get_checkpoints_dir, get_infer_dir,
                                 get_infer_log, get_train_log_dir,
                                 get_train_log_file, get_train_root_dir,
                                 get_val_dir, get_val_log,
                                 load_infer_symbols_map, load_infer_text,
                                 load_speakers_json, load_symbol_converter,
                                 load_trainset, load_valset,
                                 save_infer_alignments_sentence_plot,
                                 save_infer_plot,
                                 save_infer_pre_postnet_sentence_plot,
                                 save_infer_sentence_plot,
                                 save_infer_v_alignments, save_infer_v_plot,
                                 save_infer_v_pre_post, save_infer_wav,
                                 save_infer_wav_sentence, save_speakers_json,
                                 save_symbol_converter, save_testset,
                                 save_trainset,
                                 save_val_alignments_sentence_plot,
                                 save_val_comparison, save_val_orig_plot,
                                 save_val_orig_wav, save_val_plot,
                                 save_val_pre_postnet_plot, save_val_wav,
                                 save_valset)
from src.core.common import (Language, get_basename,
                             get_custom_or_last_checkpoint)
from src.core.inference import get_logger
from src.core.inference import infer as infer_core
from src.core.inference import validate as validate_core
from src.core.pre import (PreparedDataList, SpeakersIdDict, SymbolConverter,
                          split_train_test_val)
from src.core.tacotron import continue_train as continue_train_core
from src.core.tacotron import get_train_logger
from src.core.tacotron import train as train_core


def train(base_dir: str, train_name: str, fl_name: str, warm_start_model: str = "", test_size: float = 0.01, validation_size: float = 0.05, hparams = "", split_seed: int = 1234, weight_map_model: str = "", weight_map_model_symbols: str = "", weight_map_mode: str = "", weight_map: str = ""):
  wholeset = load_filelist(base_dir, fl_name)
  trainset, testset, valset = split_train_test_val(wholeset, test_size=test_size, val_size=validation_size, seed=split_seed, shuffle=True)
  save_trainset(base_dir, train_name, trainset)
  save_testset(base_dir, train_name, testset)
  save_valset(base_dir, train_name, valset)

  symbols_conv = load_filelist_symbol_converter(base_dir, fl_name)
  save_symbol_converter(base_dir, train_name, symbols_conv)

  speakers = load_filelist_speakers_json(base_dir, fl_name)
  save_speakers_json(base_dir, train_name, speakers)

  log_file = get_train_log_file(base_dir, train_name)
  reset_log(log_file)
  add_console_and_file_out_to_logger(get_train_logger(), log_file)
  # todo log map & args

  train_core(
    warm_start_model_path=warm_start_model,
    weights_path=weight_map,
    custom_hparams=hparams,
    logdir=get_train_log_dir(base_dir, train_name),
    n_symbols=symbols_conv.get_symbol_ids_count(),
    n_speakers=len(speakers),
    trainset=trainset,
    valset=valset,
    save_checkpoint_dir=get_checkpoints_dir(base_dir, train_name),
    save_checkpoint_log_dir=get_train_log_dir(base_dir, train_name)
  )

def continue_train(base_dir: str, train_name: str, hparams):
  assert os.path.isdir(get_train_root_dir(base_dir, train_name, create=False))

  symbols_conv = load_symbol_converter(base_dir, train_name)
  speakers = load_speakers_json(base_dir, train_name)
  add_console_and_file_out_to_logger(get_train_logger(), get_train_log_file(base_dir, train_name))

  continue_train_core(
    custom_hparams=hparams,
    logdir=get_train_log_dir(base_dir, train_name),
    n_symbols=symbols_conv.get_symbol_ids_count(),
    n_speakers=len(speakers),
    trainset=load_trainset(base_dir, train_name),
    valset=load_valset(base_dir, train_name),
    save_checkpoint_dir=get_checkpoints_dir(base_dir, train_name),
    save_checkpoint_log_dir=get_train_log_dir(base_dir, train_name)
  )
  
def validate(base_dir: str, train_name: str, entry_id: int, waveglow: str, custom_checkpoint: int = 0, sigma: float = 0.666, denoiser_strength: float = 0.01, sampling_rate: float = 22050):
  logger = get_logger()
  checkpoint_path, iteration = get_custom_or_last_checkpoint(get_checkpoints_dir(base_dir, train_name), custom_checkpoint)
  val_dir = get_val_dir(base_dir, train_name, entry_id, iteration)
  add_console_and_file_out_to_logger(logger, get_val_log(val_dir))
  val = load_valset(base_dir, train_name)
  entry = val.get_entry(entry_id)

  wav, mel_outputs, mel_outputs_postnet, alignments, orig_mel = validate_core(
    entry=entry,
    taco_path=checkpoint_path,
    waveglow_path=waveglow,
    denoiser_strength=denoiser_strength,
    sigma=sigma,
    conv=load_symbol_converter(base_dir, train_name),
    n_speakers=len(load_speakers_json(base_dir, train_name))
  )

  save_val_wav(val_dir, sampling_rate, wav)
  save_val_plot(val_dir, mel_outputs)
  save_val_orig_wav(val_dir, entry.wav_path)
  save_val_orig_plot(val_dir, orig_mel)
  save_val_pre_postnet_plot(val_dir, mel_outputs_postnet)
  save_val_alignments_sentence_plot(val_dir, alignments)
  save_val_comparison(val_dir)

  logger.info(f"Saved output to {val_dir}")


def infer(base_dir: str, train_name: str, text: str, lang: int, speaker_id: int, waveglow: str, ignore_tones: bool = False, ignore_arcs: bool = True, symbols_map: str = "", hparams: str = "", custom_checkpoint: int = 0, sentence_pause_s: float = 0.5, sigma: float = 0.666, denoiser_strength: float = 0.01, sampling_rate: float = 22050, analysis: bool = True, ipa: bool = True):
  logger = get_logger()
  input_name = get_basename(text)
  checkpoint_path, iteration = get_custom_or_last_checkpoint(get_checkpoints_dir(base_dir, train_name), custom_checkpoint)
  infer_dir = get_infer_dir(base_dir, train_name, input_name, iteration, speaker_id)
  add_console_and_file_out_to_logger(logger, get_infer_log(infer_dir))
  
  wav, wav_mel, full = infer_core(
    taco_path=checkpoint_path,
    waveglow_path=waveglow,
    conv=load_symbol_converter(base_dir, train_name),
    lines=load_infer_text(text),
    n_speakers=len(load_speakers_json(base_dir, train_name)),
    speaker_id=speaker_id,
    sentence_pause_s=sentence_pause_s,
    sigma=sigma,
    denoiser_strength=denoiser_strength,
    sampling_rate=sampling_rate,
    ipa=ipa,
    ignore_tones=ignore_tones,
    ignore_arcs=ignore_arcs,
    subset_id=0,
    lang=lang,
    symbols_map=load_infer_symbols_map(symbols_map) if symbols_map else None
  )

  save_infer_wav(infer_dir, sampling_rate, wav)
  save_infer_plot(infer_dir, wav_mel)

  if analysis:
    logger.info("Analysing...")
    for sentence_nr, mels, sent_wav in tqdm(full):
      mel_outputs, mel_outputs_postnet, alignments = mels
      save_infer_wav_sentence(infer_dir, sentence_nr, sampling_rate, sent_wav)
      save_infer_sentence_plot(infer_dir, sentence_nr, mel_outputs)
      save_infer_pre_postnet_sentence_plot(infer_dir, sentence_nr, mel_outputs_postnet)
      save_infer_alignments_sentence_plot(infer_dir, sentence_nr, alignments)
    sentence_ids = [x[0] for x in full]
    save_infer_v_plot(infer_dir, sentence_ids)
    save_infer_v_pre_post(infer_dir, sentence_ids)
    save_infer_v_alignments(infer_dir, sentence_ids)
    
  logger.info(f"Saved output to {infer_dir}")

if __name__ == "__main__":
  mode = 4
  if mode == 1:
    train(
      base_dir="/datasets/models/taco2pt_v2",
      train_name="debug_new",
      fl_name="thchs",
      hparams="batch_size=17,iters_per_checkpoint=5,epochs_per_checkpoint=1,cache_mels=False"
    )
  elif mode == 2:
    continue_train(
      base_dir="/datasets/models/taco2pt_v2",
      train_name="debug_new",
      hparams="batch_size=17,iters_per_checkpoint=5,epochs_per_checkpoint=1,cache_mels=False"
    )
  elif mode == 3:
    validate(
      base_dir="/datasets/models/taco2pt_v2",
      train_name="debug_new",
      entry_id=6,
      waveglow="/datasets/models/pretrained/waveglow_256channels_universal_v5_out.pt",
    )
  elif mode == 4:
    infer(
      base_dir="/datasets/models/taco2pt_v2",
      train_name="debug_new",
      text="examples/chn/north.txt",
      lang=Language.CHN,
      speaker_id=0,
      waveglow="/datasets/models/pretrained/waveglow_256channels_universal_v5_out.pt",
    )
