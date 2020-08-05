from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.common.train_log import log
from src.pre.wav_pre_io import parse_data, WavData, WavDataList
from src.waveglow.prepare_ds_io import (save_testset, save_trainset,
                                        save_validationset, save_wholeset, PreparedData, PreparedDataList)


def prepare(base_dir: str, training_dir_path: str, wav_ds_name: str, test_size: float, validation_size: float, seed: int):
  wholeset: PreparedDataList = []
  print("Reading wavs...")
  wav_data = parse_data(base_dir, wav_ds_name)
  values: WavData
  for values in tqdm(wav_data):
    prepared_data = PreparedData(values.basename, values.wav, values.duration)
    wholeset.append(prepared_data)

  trainset = wholeset
  print("Splitting datasets...")
  testset = []
  create_testset = test_size > 0
  if create_testset:
    trainset, testset = train_test_split(trainset, test_size=test_size, random_state=seed)
    save_testset(training_dir_path, testset)
  else:
    log(training_dir_path, "Create no testset.")

  valset = []
  create_valset = validation_size > 0
  if create_valset:
    trainset, valset = train_test_split(trainset, test_size=validation_size, random_state=seed)
    save_validationset(training_dir_path, valset)
  else:
    log(training_dir_path, "Create no valset.")
  
  save_trainset(training_dir_path, trainset)
  save_wholeset(training_dir_path, wholeset)
  log(training_dir_path, "Done.")

if __name__ == "__main__":
  prepare(
    base_dir = '/datasets/models/taco2pt_v2',
    training_dir_path = '/datasets/models/taco2pt_v2/wg_debug',
    wav_ds_name = 'ljs_22050kHz',
    test_size = 0.001,
    validation_size = 0.01,
    seed = 1234,
  )
