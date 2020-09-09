import logging
import os

formatter = logging.Formatter(
  '[%(asctime)s.%(msecs)03d] (%(levelname)s) %(message)s',
  datefmt='%Y/%m/%d %H:%M:%S'
)


def get_default_logger():
  return logging.getLogger("default")


def init_logger(logger: logging.Logger = get_default_logger()):
  root_logger = logging.getLogger()
  root_logger.setLevel(logging.DEBUG)
  # disable is required (don't know why) because otherwise DEBUG messages would be ignored!
  logger.manager.disable = logging.NOTSET

  # to disable double logging
  logger.propagate = False

  # take it from the above logger (root)
  logger.setLevel(logging.DEBUG)

  for h in logger.handlers:
    logger.removeHandler(h)

  return logger


def add_console_out_to_logger(logger: logging.Logger = get_default_logger()):
  console_handler = logging.StreamHandler()
  console_handler.setLevel(logging.NOTSET)
  console_handler.setFormatter(formatter)
  logger.addHandler(console_handler)
  logger.debug("init console logger")


def add_file_out_to_logger(logger: logging.Logger = get_default_logger(), log_file_path: str = "/tmp/log.txt"):
  fh = logging.FileHandler(log_file_path)
  fh.setLevel(logging.INFO)
  fh.setFormatter(formatter)
  logger.addHandler(fh)
  logger.debug(f"init fh logger to {log_file_path}")


def reset_file_log(log_file_path: str):
  if os.path.isfile(log_file_path):
    os.remove(log_file_path)


if __name__ == "__main__":
  test_logger = logging.getLogger("test")

  add_console_out_to_logger(test_logger)
