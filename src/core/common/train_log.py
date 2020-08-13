import datetime
import os

from src.core.common.utils import get_subdir

def log(log_path: str, msg: str):
  msg_with_timepoint = "[{}] {}".format(datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"), msg)
  with open(log_path, 'a', encoding='utf-8') as f:
    print(msg_with_timepoint)
    f.write(msg_with_timepoint + '\n')

def reset_log(log_path: str):
  with open(log_path, 'w', encoding='utf-8') as f:
    f.write('')
