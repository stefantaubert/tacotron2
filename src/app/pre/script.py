from src.app.pre.ds import (preprocess_ljs, preprocess_thchs,
                            preprocess_thchs_kaldi)
from src.app.pre.mel import preprocess_mels
from src.app.pre.prepare import prepare_ds
from src.app.pre.text import (preprocess_text, text_convert_to_ipa,
                              text_normalize)
from src.app.pre.wav import (preprocess_wavs, wavs_normalize,
                             wavs_remove_silence, wavs_upsample)


def prepare_thchs(base_dir: str):
  preprocess_thchs(
    base_dir=base_dir,
    path="/datasets/thchs_wav",
    auto_dl=True,
    ds_name="thchs",
  )

  preprocess_text(
    base_dir=base_dir,
    ds_name="thchs",
    text_name="chn",
  )

  text_convert_to_ipa(
    base_dir=base_dir,
    ds_name="thchs",
    orig_text_name="chn",
    dest_text_name="ipa",
    ignore_arcs=True
  )

  preprocess_wavs(
    base_dir=base_dir,
    ds_name="thchs",
    wav_name="16000Hz",
  )

  wavs_normalize(
    base_dir=base_dir,
    ds_name="thchs",
    orig_wav_name="16000Hz",
    dest_wav_name="16000Hz_norm",
  )

  wavs_remove_silence(
    base_dir=base_dir,
    ds_name="thchs",
    orig_wav_name="16000Hz_norm",
    dest_wav_name="16000Hz_norm_wo_sil",
    chunk_size=5,
    threshold_start=-20,
    threshold_end=-30,
    buffer_start_ms=100,
    buffer_end_ms=150
  )

  wavs_upsample(
    base_dir=base_dir,
    ds_name="thchs",
    orig_wav_name="16000Hz_norm_wo_sil",
    dest_wav_name="22050Hz_norm_wo_sil",
    rate=22050,
  )

  preprocess_mels(
    base_dir=base_dir,
    ds_name="thchs",
    wav_name="22050Hz_norm_wo_sil"
  )

  prepare_ds(
    base_dir=base_dir,
    prep_name="thchs_ipa",
    ds_speakers=[("thchs", "all")],
    ds_text_audio=[("thchs", "ipa", "22050Hz_norm_wo_sil")]
  )


def prepare_thchs_kaldi(base_dir: str):
  preprocess_thchs_kaldi(
    base_dir=base_dir,
    path="/datasets/THCHS-30",
    ds_name="thchs_kaldi",
    auto_dl=True,
  )


def prepare_ljs(base_dir: str):
  preprocess_ljs(
    base_dir=base_dir,
    path="/datasets/LJSpeech-1.1",
    ds_name="ljs",
    auto_dl=True,
  )

  preprocess_text(
    base_dir=base_dir,
    ds_name="ljs",
    text_name="en",
  )

  text_normalize(
    base_dir=base_dir,
    ds_name="ljs",
    orig_text_name="en",
    dest_text_name="en_norm",
  )

  text_convert_to_ipa(
    base_dir=base_dir,
    ds_name="ljs",
    orig_text_name="en_norm",
    dest_text_name="ipa_norm",
    ignore_tones=True,
    ignore_arcs=True
  )

  preprocess_wavs(
    base_dir=base_dir,
    ds_name="ljs",
    wav_name="22050kHz",
  )

  preprocess_mels(
    base_dir=base_dir,
    ds_name="ljs",
    wav_name="22050kHz"
  )

  prepare_ds(
    base_dir=base_dir,
    prep_name="ljs_ipa",
    ds_speakers=[("ljs", "all")],
    ds_text_audio=[("ljs", "ipa_norm", "22050kHz")]
  )

if __name__ == "__main__":
  run_all = True
  mode = 1

  base_dir = "/datasets/models/taco2pt_v5"

  if run_all or mode == 1:
    prepare_ljs(base_dir)

  if run_all or mode == 2:
    prepare_thchs(base_dir)

  if run_all or mode == 3:
    prepare_ds(
      base_dir=base_dir,
      prep_name="thchs_ljs_ipa",
      ds_speakers=[("thchs", "all"), ("ljs", "all")],
      ds_text_audio=[
        ("thchs", "ipa", "22050Hz_norm_wo_sil"),
        ("ljs", "ipa_norm", "22050kHz"),
      ]
    )
  # if run_all or mode == 2:
  #   prepare_thchs_kaldi()
