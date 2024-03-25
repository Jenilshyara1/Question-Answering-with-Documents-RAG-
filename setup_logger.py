import os
import sys
import logging

logging_str= "%(asctime)s: %(levelname)s: %(module)s: %(message)s"

log_dir="logs"
log_filepath=os.path.join(log_dir,"app.log")
os.makedirs(log_dir,exist_ok=True)

def init_logger():
    logging.basicConfig(
        level=logging.INFO,
        format=logging_str,
        handlers=[
            logging.FileHandler(log_filepath),
            logging.StreamHandler(sys.stdout)
        ]
    )

