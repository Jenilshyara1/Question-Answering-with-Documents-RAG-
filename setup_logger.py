import os
import logging
from logging.handlers import RotatingFileHandler


def init_logger(loglevel, log_folder_path, log_file_path):
    if not os.path.exists(log_folder_path):
        os.makedirs(log_folder_path)
    logger = logging.getLogger()
    if loglevel == "info":
        logger.setLevel(logging.INFO)
    elif loglevel == "debug":
        logger.setLevel(logging.DEBUG)
    format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(format)
    logger.addHandler(stream_handler)
    file_handler = RotatingFileHandler(
        os.path.join(log_folder_path, log_file_path), maxBytes=2000, backupCount=5
    )
    file_handler.setFormatter(format)
    logger.addHandler(file_handler)
    return logger
