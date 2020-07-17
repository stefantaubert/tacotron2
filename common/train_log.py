import os
from script_paths import get_log_dir, log_train_file_name
import datetime

def log(training_dir_path: str, msg: str):
  log_path = os.path.join(get_log_dir(training_dir_path), log_train_file_name)
  msg_with_timepoint = "[{}] {}".format(datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"), msg)
  with open(log_path, 'a', encoding='utf-8') as f:
    print(msg_with_timepoint)
    f.write(msg_with_timepoint + '\n')

def reset_log(training_dir_path: str):
  log_path = os.path.join(get_log_dir(training_dir_path), log_train_file_name)
  with open(log_path, 'w', encoding='utf-8') as f:
    f.write('')
