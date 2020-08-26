import os
from typing import List, Tuple

from src.app.pre.ds import (preprocess_ljs, preprocess_thchs,
                            preprocess_thchs_kaldi)
from src.app.pre.mel import preprocess_mels
from src.app.pre.prepare import prepare_ds
from src.app.pre.text import (preprocess_text, text_convert_to_ipa,
                              text_normalize)
from src.app.pre.wav import (preprocess_wavs, wavs_normalize,
                             wavs_remove_silence, wavs_upsample)


def prepare_thchs():
  preprocess_thchs(
    path="/datasets/thchs_wav",
    base_dir="/datasets/models/taco2pt_v4",
    ds_name="thchs",
    auto_dl=True,
  )

  preprocess_text(
    base_dir="/datasets/models/taco2pt_v4",
    ds_name="thchs",
    text_name="chn",
  )

  text_convert_to_ipa(
    base_dir="/datasets/models/taco2pt_v4",
    ds_name="thchs",
    orig_text_name="chn",
    dest_text_name="ipa",
    ignore_tones=False,
    ignore_arcs=True,
  )

  preprocess_wavs(
    base_dir="/datasets/models/taco2pt_v4",
    ds_name="thchs",
    wav_name="16000Hz",
  )

  wavs_normalize(
    base_dir="/datasets/models/taco2pt_v4",
    ds_name="thchs",
    orig_wav_name="16000Hz",
    dest_wav_name="16000Hz_normalized",
  )

  wavs_remove_silence(
    base_dir="/datasets/models/taco2pt_v4",
    ds_name="thchs",
    orig_wav_name="16000Hz_normalized",
    dest_wav_name="16000Hz_normalized_nosil",
    threshold_start = -20,
    threshold_end = -30,
    chunk_size = 5,
    buffer_start_ms = 100,
    buffer_end_ms = 150
  )

  wavs_upsample(
    base_dir="/datasets/models/taco2pt_v4",
    ds_name="thchs",
    orig_wav_name="16000Hz_normalized_nosil",
    dest_wav_name="22050Hz_normalized_nosil",
    rate=22050,
  )

  preprocess_mels(
    base_dir="/datasets/models/taco2pt_v4",
    ds_name="thchs",
    wav_name="22050Hz_normalized_nosil",
    custom_hparams="",
  )

  prepare_ds(
    base_dir="/datasets/models/taco2pt_v4",
    prep_name="thchs",
    ds_speakers=[("thchs", "all")],
    ds_text_audio=[("thchs", "ipa", "22050Hz_normalized_nosil")]
  )

def prepare_thchs_kaldi():
  preprocess_thchs_kaldi(
    base_dir="/datasets/models/taco2pt_v4",
    path="/datasets/THCHS-30",
    ds_name="thchs_kaldi",
    auto_dl=True,
  )

def prepare_ljs():
  preprocess_ljs(
    base_dir="/datasets/models/taco2pt_v4",
    path="/datasets/LJSpeech-1.1",
    ds_name="ljs",
    auto_dl=True,
  )

  preprocess_text(
    base_dir="/datasets/models/taco2pt_v4",
    ds_name="ljs",
    text_name="en",
  )

  text_normalize(
    base_dir="/datasets/models/taco2pt_v4",
    ds_name="ljs",
    orig_text_name="en",
    dest_text_name="en_norm",
  )

  text_convert_to_ipa(
    base_dir="/datasets/models/taco2pt_v4",
    ds_name="ljs",
    orig_text_name="en_norm",
    dest_text_name="ipa_norm",
    ignore_tones=True,
    ignore_arcs=True,
  )

  preprocess_wavs(
    base_dir="/datasets/models/taco2pt_v4",
    ds_name="ljs",
    wav_name="22050kHz",
  )

  preprocess_mels(
    base_dir="/datasets/models/taco2pt_v4",
    ds_name="ljs",
    wav_name="22050kHz",
    custom_hparams="",
  )

  prepare_ds(
    base_dir="/datasets/models/taco2pt_v4",
    prep_name="ljs_ipa",
    ds_speakers=[("ljs", "all")],
    ds_text_audio=[("ljs", "ipa_norm", "22050kHz")]
  )

if __name__ == "__main__":
  run_all = False
  mode = 1
  
  if run_all or mode == 1:
    prepare_thchs()
  
  if run_all or mode == 2:
    prepare_thchs_kaldi()

  if run_all or mode == 3:
    prepare_ljs()
