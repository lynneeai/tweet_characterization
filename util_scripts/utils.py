import re
import logging
import time
from datetime import datetime

import pycld2 as cld2
from urllib.parse import urlparse
from tqdm import tqdm


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
    

def boolean_string(s):
    if s.lower() not in {"false", "true", "t", "f", "yes", "no", "y", "n"}:
        raise ValueError('Not a valid boolean string')
    return s.lower() == "true" or s.lower() == "t" or s.lower() == "yes" or s.lower() == "y"


def remove_url(txt):
    return re.sub(r'https?://\S+', '', txt, flags=re.MULTILINE)


def is_english(text):
    try:
        _, _, _, detected_language = cld2.detect(text, returnVectors=True)
        all_english = True
        for lang in detected_language:
            lang = lang[-1]
            if lang != "en":
                all_english = False
                break
    except:
        print(text)
        return False
    return all_english


def count_file_lines(fname):
    """
    Counts number of lines in file
    """
    lines = 0
    with open(fname, "r") as infile:
        for line in infile:
            lines += 1
    return lines