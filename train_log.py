import os
from paths import get_log_dir, log_train_file_name

def log(training_dir_path: str, msg: str):
  log_path = os.path.join(get_log_dir(training_dir_path), log_train_file_name)
  with open(log_path, 'a', encoding='utf-8') as f:
    print(msg)
    f.write(msg + '\n')

def reset_log(training_dir_path: str):
  log_path = os.path.join(get_log_dir(training_dir_path), log_train_file_name)
  with open(log_path, 'w', encoding='utf-8') as f:
    f.write('')
