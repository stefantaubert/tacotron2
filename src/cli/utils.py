import logging
import os

def add_console_and_file_out_to_logger(logger: logging.Logger, log_file_path: str = ""):
  logger.propagate = False
  logger.setLevel(logging.DEBUG)
  formatter = logging.Formatter(
    '[%(asctime)s] (%(levelname)s) %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S'
  )

  console_handler = logging.StreamHandler()
  console_handler.setLevel(logging.DEBUG)
  console_handler.setFormatter(formatter)
  logger.addHandler(console_handler)
  logger.info("init console logger")
  
  if log_file_path:
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.info("init fh logger")

def reset_log(log_file_path: str):
  if os.path.isfile(log_file_path):
    os.remove(log_file_path)