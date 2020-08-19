from src.app.utils import add_console_and_file_out_to_logger, reset_log

from src.app.pre import preprocess_ljs, preprocess_thchs, preprocess_thchs_kaldi, preprocess_mels, prepare_ds, preprocess_text, text_normalize, text_convert_to_ipa, preprocess_wavs, wavs_normalize, wavs_remove_silence, wavs_upsample

from src.app.tacotron import train as taco_train, continue_train as taco_continue_train, infer as taco_infer, validate as taco_validate

from src.app.waveglow import train as wg_train, continue_train as wg_continue_train, infer as wg_infer, validate as wg_validate, dl_pretrained as wg_dl_pretrained

from src.app.io import get_train_root_dir, get_train_logs_dir, get_train_log_file, get_checkpoints_dir, save_trainset, load_trainset, save_testset, load_testset, save_valset, load_valset, get_inference_root_dir, get_infer_log, save_infer_wav, save_infer_plot, get_val_dir, save_val_plot, save_val_orig_plot, save_val_comparison, save_val_wav, save_val_orig_wav, get_val_log
