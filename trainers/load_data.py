import csv
import os
import sys
import logging
import random
import math
from torch.utils.data import DataLoader

"""Solve import issue"""
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = f"{current_file_dir}/.."
sys.path.append(current_file_dir)
sys.path.append(project_root_dir)
"""------------------"""

from util_scripts.utils import init_logger


def create_partitions(config, tsv_fieldnames):
    log_filename = os.path.basename(__file__).split(".")[0]
    init_logger(config.LOGS_ROOT, config.OUTPUT_FILES_NAME)
    logger = logging.getLogger(log_filename)
        
    logger.info("Creating partitions...")
    
    train_split=config.TRAIN_SPLIT
    dev_split=config.DEV_SPLIT
    
    tweet_obj_list = []
    with open(f"{config.TSV_ROOT}/tweets.tsv", "r") as infile:
        tsv_reader = csv.DictReader(infile, delimiter="\t")
        for row in tsv_reader:
            tweet_obj_list.append(dict(row))
    
    # randomly generate train dev test sets
    total_tweets = len(tweet_obj_list)
    train_len = math.floor(total_tweets * train_split)
    dev_len = math.floor(total_tweets * dev_split)
    test_len = total_tweets - train_len - dev_len
    
    all_idx = [i for i in range(total_tweets)]
    random.shuffle(all_idx)
    
    train_idx = all_idx[:train_len]
    dev_idx = all_idx[train_len:train_len+dev_len]
    test_idx = all_idx[train_len+dev_len:]
    assert len(train_idx) == train_len
    assert len(dev_idx) == dev_len
    assert len(test_idx) == test_len
    
    # write to tsv files
    for batch_name, batch_idx in [("train", train_idx), ("dev", dev_idx), ("test", test_idx)]:
        with open(f"{config.TSV_ROOT}/{batch_name}.tsv", "w") as outfile:
            tsv_writer = csv.DictWriter(outfile, delimiter="\t", fieldnames=tsv_fieldnames)
            tsv_writer.writeheader()
            for idx in batch_idx:
                tweet_obj = tweet_obj_list[idx]
                tweet_obj["image_file"] = f"{project_root_dir}/{tweet_obj['image_file']}"
                tsv_writer.writerow(tweet_obj)
                outfile.flush()
                

def load_dataloaders(config, dataset):
    batch_size=config.BATCH_SIZE
    tsv_root=config.TSV_ROOT
    
    train_dataset = dataset(f"{tsv_root}/train.tsv")
    dev_dataset = dataset(f"{tsv_root}/dev.tsv")
    test_dataset = dataset(f"{tsv_root}/test.tsv")
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    return train_dataloader, dev_dataloader, test_dataloader
