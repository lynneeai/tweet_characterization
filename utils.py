import logging
import os
import sys
import time
from datetime import datetime

from urllib.parse import urlparse
from tqdm import tqdm


"""Solve import issue"""
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = current_file_dir
sys.path.append(current_file_dir)
sys.path.append(project_root_dir)
"""------------------"""


def program_sleep(sec, print_progress=True):
    if print_progress:
        trange = tqdm(
            range(sec),
            bar_format="sleeping for {n_fmt}/{total_fmt} seconds...",
            leave=False,
        )
        for _ in trange:
            time.sleep(1)
        trange.close()
        print(f"Done sleeping! Slept {sec} seconds!")
    else:
        time.sleep(sec)
        

def init_logger(log_folder, log_filename, timestamp=True):
    """define logging if not already defined"""
    if not logging.getLogger().handlers:
        log_format = "%(asctime)s  %(name)8s  %(levelname)5s  %(message)s"
        log_formatter = logging.Formatter(log_format)
        logging.getLogger().setLevel(logging.INFO)
        # file handler
        if timestamp:
            now_dt = datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
            log_filename = f"{log_folder}/{log_filename}_{now_dt}.log"
        else:
            log_filename = f"{log_folder}/{log_filename}.log"
        fh = logging.FileHandler(filename=log_filename, mode="w")
        fh.setLevel(logging.INFO)
        fh.setFormatter(log_formatter)
        logging.getLogger().addHandler(fh)
        # stream handler
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(log_formatter)
        logging.getLogger().addHandler(console)
    """---------------------------------"""